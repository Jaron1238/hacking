# data/capture.py

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypedDict

from tqdm import tqdm

# Relative Imports
from .. import config
from ..storage import database
from ..storage.data_models import WifiEvent
from ..analysis.deep_packet_inspection import DeepPacketInspector

logger = logging.getLogger(__name__)


from scapy.all import Dot11, Dot11Beacon, Dot11Elt, PcapReader, RadioTap, wrpcap
from scapy.layers.dhcp import BOOTP, DHCP
from scapy.layers.dns import DNSQR
from scapy.layers.l2 import ARP, LLC, SNAP


def _extract_seq(dot11) -> Optional[int]:
    try:
        sc = getattr(dot11, "SC", 0)
        if sc is None:
            return None
        return int(sc) >> 4
    except (TypeError, ValueError):
        return None


def _collect_ies_from_pkt(pkt) -> Tuple[Dict[int, List[str]], List[int]]:
    """Sammelt IEs und deren Reihenfolge."""
    ies: Dict[int, List[str]] = defaultdict(list)
    ie_order: List[int] = []
    if Dot11Elt is None:
        return {}, []
    try:
        # Starte die Suche nach dem ersten Dot11Elt Layer
        current_layer = pkt.getlayer(Dot11Elt)
        while current_layer:
            # KORREKTUR: Stelle sicher, dass es sich um ein echtes Dot11Elt-Objekt handelt
            # und nicht um eine andere Schicht wie Dot11Beacon.
            if isinstance(current_layer, Dot11Elt):
                if hasattr(current_layer, "ID"):
                    info_bytes = (
                        bytes(current_layer.info)
                        if hasattr(current_layer, "info")
                        else b""
                    )
                    ies[int(current_layer.ID)].append(info_bytes.hex())
                    ie_order.append(int(current_layer.ID))

            # Gehe zum nächsten Layer in der Payload
            current_layer = current_layer.payload.getlayer(Dot11Elt)
    except Exception:
        pass
    return dict(ies), ie_order


def packet_to_event(pkt) -> Optional[WifiEvent]:
    ts = float(pkt.time) if hasattr(pkt, "time") else time.time()
    if not pkt.haslayer(Dot11):
        return None

    rt_fields = {}
    if RadioTap and pkt.haslayer(RadioTap):
        rt = pkt.getlayer(RadioTap)

        rssi = getattr(rt, "dBm_AntSignal", None)
        if rssi is not None:
            rt_fields["rssi"] = int(rssi)

        noise = getattr(rt, "dBm_AntNoise", None)
        if noise is not None:
            rt_fields["noise"] = int(noise)

        mcs_index = getattr(rt, "MCS_index", None)
        if mcs_index is not None:
            rt_fields["mcs_index"] = int(mcs_index)

        freq = getattr(rt, "ChannelFrequency", None)
        if freq is not None:
            freq = int(freq)
            if 2412 <= freq <= 2484:
                rt_fields["channel"] = (freq - 2407) // 5
            elif 5180 <= freq <= 5825:
                rt_fields["channel"] = (freq - 5000) // 5

        flags = getattr(rt, "Flags", None)
        if flags is not None and hasattr(flags, "FCS_err") and flags.FCS_err:
            rt_fields["fcs_error"] = True

    dot = pkt.getlayer(Dot11)

    event = WifiEvent(ts=ts, type="", **rt_fields)

    if dot.type == 0:  # Management Frame
        ies, ie_order = _collect_ies_from_pkt(pkt)
        event["ies"] = ies
        event["ie_order_hash"] = hash(tuple(ie_order))

        if dot.subtype in (8, 5):  # Beacon or Probe Response
            event["type"] = "beacon" if dot.subtype == 8 else "probe_resp"
            event["bssid"] = dot.addr3

            if dot.haslayer(Dot11Beacon):
                bcn = dot.getlayer(Dot11Beacon)
                if hasattr(bcn, "beacon_interval"):
                    event["beacon_interval"] = int(bcn.beacon_interval)
                if hasattr(bcn, "cap"):
                    event["cap"] = int(bcn.cap)

            try:
                ssid = (
                    pkt.info.decode(errors="ignore")
                    if hasattr(pkt, "info")
                    else "<hidden>"
                )
                event["ssid"] = ssid or "<hidden>"
            except Exception:
                event["ssid"] = "<binary>"

            return event

        elif dot.subtype == 4:  # Probe Request
            event["type"] = "probe_req"
            event["client"] = dot.addr2
            probes = [
                bytes.fromhex(h).decode(errors="ignore") or "<broadcast>"
                for h in ies.get(0, [])
            ]
            ies["probes"] = probes
            return event

    elif dot.type == 2:  # Data Frame
        event["type"] = "data"
        event["client"] = dot.addr2
        event["bssid"] = dot.addr1
        event["seq"] = _extract_seq(dot)
        event["is_powersave"] = bool(dot.FCfield & 0x10)

        if pkt.haslayer(DNSQR) and pkt.dport == 53:
            try:
                dns_name = pkt[DNSQR].qname.decode(errors="ignore")
                # Entferne den abschließenden Punkt von DNS-Namen
                event["dns_query"] = dns_name.rstrip('.')
            except Exception:
                pass
        elif pkt.haslayer(DHCP):
            try:
                dhcp_options = {
                    opt[0]: opt[1]
                    for opt in pkt[DHCP].options
                    if isinstance(opt, tuple)
                }
                if dhcp_options.get(53) in [3, 5] and 12 in dhcp_options:
                    event["hostname"] = dhcp_options[12].decode(errors="ignore")
            except Exception:
                pass
        elif pkt.haslayer(ARP) and pkt[ARP].op == 2:
            event["arp_mac"] = pkt[ARP].hwsrc
            event["arp_ip"] = pkt[ARP].psrc

        return event

    return None


class ChannelHopper(threading.Thread):
    def __init__(self, iface: str, channels: List[int], sleep_interval: float):
        super().__init__(daemon=True, name="ChannelHopper")
        self.iface, self.channels, self.sleep_interval = iface, channels, sleep_interval
        self.stop_event = threading.Event()
        logger.info(
            "Channel Hopper initialisiert für %s mit %d Kanälen.", iface, len(channels)
        )

    def run(self):
        idx = 0
        if not self.channels:
            logger.warning(
                "Channel Hopper hat keine Kanäle zum Springen und wird beendet."
            )
            return

        while not self.stop_event.is_set():
            channel = self.channels[idx]

            if 1 <= channel <= 14:
                bw = config.BANDWIDTH_2_4GHZ
            else:
                bw = config.BANDWIDTH_5GHZ

            try:
                command = ["iw", "dev", self.iface, "set", "channel", str(channel), bw]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    logger.debug(
                        f"Setzen des Kanals mit Bandbreite {bw} fehlgeschlagen, versuche Fallback..."
                    )
                    fallback_command = [
                        "iw",
                        "dev",
                        self.iface,
                        "set",
                        "channel",
                        str(channel),
                    ]
                    result = subprocess.run(
                        fallback_command, check=True, capture_output=True, text=True
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e_fallback:
                    logger.error(
                        "Fehler beim Setzen des Kanals auf %s: %s",
                        self.iface,
                        e_fallback,
                    )
                    time.sleep(5)
                except Exception as e:
                    logger.error("Unerwarteter Fehler beim Channel-Hopping: %s", e)
                    time.sleep(1)

            idx = (idx + 1) % len(self.channels)
            time.sleep(self.sleep_interval)

    def stop(self):
        self.stop_event.set()


def packet_parser_worker(
    pcap_filename_queue: mp.Queue, db_queue: mp.Queue, live_queue: Optional[mp.Queue]
):
    while True:
        try:
            filename = pcap_filename_queue.get()
            if filename is None:
                break
            with PcapReader(filename) as pcap_reader:
                for pkt in pcap_reader:
                    if ev := packet_to_event(pkt):
                        db_queue.put(ev)
                        if live_queue:
                            try:
                                live_queue.put_nowait(ev)
                            except queue.Full:
                                pass
            os.remove(filename)
        except Exception as e:
            logger.debug(f"Fehler im Parser-Worker: {e}")


def packet_reader_thread(
    tcpdump_proc, pcap_filename_queue: mp.Queue, pcap_buff, stop_event, pcap_out
):
    try:
        pcap_header = tcpdump_proc.stdout.read(24)
        if not pcap_header:
            logger.warning(
                "tcpdump hat keinen Pcap-Header gesendet. Beende Reader-Thread."
            )
            return
        if pcap_out:
            pcap_buff.append(pcap_header)
        chunk_size = 1024 * 256
        while not stop_event.is_set():
            pkt_data_chunk = tcpdump_proc.stdout.read(chunk_size)
            if not pkt_data_chunk:
                break
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pcap", prefix="capture_chunk_"
            ) as tmp_file:
                tmp_file.write(pcap_header)
                tmp_file.write(pkt_data_chunk)
                tmp_filename = tmp_file.name
            pcap_filename_queue.put(tmp_filename)
            if pcap_out:
                pcap_buff.append(pkt_data_chunk)
    except Exception as e:
        logger.error(f"Fehler im tcpdump-Reader-Thread: {e}")
    finally:
        logger.info("packet_reader_thread beendet.")


def stderr_drainer_thread(proc, stop_event):
    while not stop_event.is_set():
        line = proc.stderr.readline()
        if not line:
            break
        logger.debug(f"[tcpdump stderr] {line.decode().strip()}")


def sniff_with_writer(
    iface: str,
    duration: int,
    db_path: str,
    pcap_out: Optional[str] = None,
    live_queue: Optional[mp.Queue] = None,
    channels_override: Optional[List[int]] = None,
):
    if PcapReader is None:
        raise RuntimeError("scapy nicht installiert")

    pcap_filename_queue = mp.Queue(maxsize=128)
    db_event_queue: mp.Queue = mp.Queue(maxsize=20000)
    num_workers = max(1, mp.cpu_count() - 1)
    workers = [
        mp.Process(
            target=packet_parser_worker,
            args=(pcap_filename_queue, db_event_queue, live_queue),
            daemon=True,
        )
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    writer = database.BatchedEventWriter(
        db_path, db_event_queue, config.DB_BATCH_SIZE, config.DB_FLUSH_INTERVAL_S
    )

    channels_to_use = (
        channels_override if channels_override is not None else config.CHANNELS_TO_HOP
    )

    hopper = ChannelHopper(iface, channels_to_use, config.CHANNEL_HOP_SLEEP_S)

    pcap_buff_bytes, threads = [], [writer, hopper]

    logger.info("Starte tcpdump als Sniffer-Backend auf %s...", iface)
    bpf_filter = "wlan type mgt or wlan type data"
    logger.info(
        f"Verwende BPF-Filter: '{bpf_filter}' zur Erfassung von Management- und Daten-Paketen."
    )
    tcpdump_cmd = ["tcpdump", "-l", "-i", iface, "-w", "-", bpf_filter]
    tcpdump_proc = subprocess.Popen(
        tcpdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    stop_event = threading.Event()
    reader = threading.Thread(
        target=packet_reader_thread,
        args=(tcpdump_proc, pcap_filename_queue, pcap_buff_bytes, stop_event, pcap_out),
        daemon=True,
        name="TcpdumpReader",
    )
    threads.append(reader)
    stderr_drainer = threading.Thread(
        target=stderr_drainer_thread,
        args=(tcpdump_proc, stop_event),
        daemon=True,
        name="StderrDrainer",
    )
    threads.append(stderr_drainer)

    for t in threads:
        t.start()

    logger.info(
        "Sniffing auf %s für %ds mit %d Workern...", iface, duration, num_workers
    )

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            if tcpdump_proc.poll() is not None:
                err = tcpdump_proc.stderr.read().decode()
                logger.error(f"tcpdump wurde unerwartet beendet. Fehlermeldung: {err}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Sniffing durch Benutzer unterbrochen.")
    finally:
        logger.info("Sniffing beendet. Stoppe Threads, Worker und tcpdump...")
        stop_event.set()
        tcpdump_proc.terminate()
        try:
            tcpdump_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tcpdump_proc.kill()

        time.sleep(0.5)
        initial_raw_size = pcap_filename_queue.qsize()
        initial_event_size = db_event_queue.qsize()

        def _monitor_progress(stop_event_progress):
            with tqdm(
                total=max(1, initial_raw_size), desc="1. Pakete parsen ", unit="file"
            ) as pbar_parse, tqdm(
                total=max(1, initial_event_size), desc="2. Events schreiben", unit="evt"
            ) as pbar_write:
                while not stop_event_progress.is_set():
                    current_parsed = max(
                        0, initial_raw_size - pcap_filename_queue.qsize()
                    )
                    pbar_parse.n = min(current_parsed, initial_raw_size)
                    pbar_parse.refresh()
                    current_written = max(
                        0, initial_event_size - db_event_queue.qsize()
                    )
                    pbar_write.n = min(current_written, initial_event_size)
                    pbar_write.refresh()
                    if (
                        pcap_filename_queue.qsize() == 0
                        and db_event_queue.qsize() == 0
                        and pbar_parse.n >= initial_raw_size
                    ):
                        break
                    time.sleep(0.1)
                pbar_parse.n = initial_raw_size
                pbar_parse.refresh()
                pbar_write.n = initial_event_size
                pbar_write.refresh()

        progress_stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=_monitor_progress, args=(progress_stop_event,), daemon=True
        )
        progress_thread.start()

        for _ in range(num_workers):
            pcap_filename_queue.put(None)
        for w in workers:
            w.join(timeout=20)
        if w.is_alive():
            w.terminate()
        for t in [writer, hopper]:
            t.stop()
        for t in threads:
            t.join(timeout=3)

        progress_stop_event.set()
        progress_thread.join(timeout=2)

        if pcap_out and pcap_buff_bytes:
            try:
                with open(pcap_out, "wb") as f:
                    for chunk in pcap_buff_bytes:
                        f.write(chunk)
                logger.info("PCAP geschrieben nach %s", pcap_out)
            except Exception as e:
                logger.exception("PCAP-Schreiben fehlgeschlagen: %s", e)
