#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Packet Inspection (DPI) Module für erweiterte Protokoll-Analyse.
Analysiert Layer 7-Protokolle wie HTTP, DNS, DHCP für besseren Informationsgewinn.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime

from scapy.all import (
    IP, IPv6, TCP, UDP, 
    Raw, DNS, DNSQR, DNSRR,
    DHCP, BOOTP, ARP
)
from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse  # <-- KORRIGIERT
from scapy.layers.inet import IPField
from scapy.layers.l2 import Ether


logger = logging.getLogger(__name__)


@dataclass
class ProtocolAnalysis:
    """Ergebnis einer Protokoll-Analyse."""
    protocol: str
    source_mac: str
    dest_mac: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    port: Optional[int] = None
    payload_size: int = 0
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DNSAnalysis:
    """DNS-Analyse-Ergebnis."""
    query_type: str
    domain: str
    response_code: Optional[int] = None
    answers: List[str] = None
    is_malicious: bool = False
    ttl: Optional[int] = None
    
    def __post_init__(self):
        if self.answers is None:
            self.answers = []


@dataclass
class HTTPAnalysis:
    """HTTP-Analyse-Ergebnis."""
    method: str
    url: str
    host: str
    user_agent: Optional[str] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    is_https: bool = False
    suspicious_patterns: List[str] = None
    
    def __post_init__(self):
        if self.suspicious_patterns is None:
            self.suspicious_patterns = []


@dataclass
class DHCPAnalysis:
    """DHCP-Analyse-Ergebnis."""
    message_type: str
    client_mac: str
    requested_ip: Optional[str] = None
    server_ip: Optional[str] = None
    hostname: Optional[str] = None
    vendor_class: Optional[str] = None
    lease_time: Optional[int] = None


class DeepPacketInspector:
    """Deep Packet Inspector für erweiterte Protokoll-Analyse."""
    
    def __init__(self):
        """Initialisiert den Deep Packet Inspector."""
        self.dns_queries = defaultdict(list)
        self.http_requests = defaultdict(list)
        self.dhcp_leases = defaultdict(list)
        self.suspicious_domains = self._load_suspicious_domains()
        self.device_fingerprints = defaultdict(dict)
        
    def _load_suspicious_domains(self) -> Set[str]:
        """Lädt Liste verdächtiger Domains."""
        # In Produktion würde das aus einer Datei oder API geladen
        return {
            "malware.com", "phishing.net", "suspicious.org",
            "botnet.io", "c2server.com"
        }
    
    def analyze_packet(self, packet) -> Optional[ProtocolAnalysis]:
        """
        Analysiert ein Paket auf Layer 7-Protokolle.
        
        Args:
            packet: Scapy-Paket
            
        Returns:
            ProtocolAnalysis oder None
        """
        try:
            # IP-Layer prüfen
            if packet.haslayer(IP):
                return self._analyze_ip_packet(packet)
            elif packet.haslayer(IPv6):
                return self._analyze_ipv6_packet(packet)
            elif packet.haslayer(ARP):
                return self._analyze_arp_packet(packet)
            
        except Exception as e:
            logger.error(f"Fehler bei Paket-Analyse: {e}")
            
        return None
    
    def _analyze_ip_packet(self, packet) -> Optional[ProtocolAnalysis]:
        """Analysiert IPv4-Pakete."""
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        # MAC-Adressen extrahieren
        src_mac = packet.src if hasattr(packet, 'src') else "unknown"
        dst_mac = packet.dst if hasattr(packet, 'dst') else "unknown"
        
        # Protokoll-spezifische Analyse
        if packet.haslayer(TCP):
            return self._analyze_tcp_packet(packet, src_mac, dst_mac, src_ip, dst_ip)
        elif packet.haslayer(UDP):
            return self._analyze_udp_packet(packet, src_mac, dst_mac, src_ip, dst_ip)
        
        return None
    
    def _analyze_tcp_packet(self, packet, src_mac: str, dst_mac: str, 
                          src_ip: str, dst_ip: str) -> Optional[ProtocolAnalysis]:
        """Analysiert TCP-Pakete."""
        tcp_layer = packet[TCP]
        port = tcp_layer.dport
        
        # HTTP-Analyse
        if port in [80, 8080, 8000]:
            http_analysis = self._analyze_http(packet)
            if http_analysis:
                return ProtocolAnalysis(
                    protocol="HTTP",
                    source_mac=src_mac,
                    dest_mac=dst_mac,
                    source_ip=src_ip,
                    dest_ip=dst_ip,
                    port=port,
                    payload_size=len(packet[Raw]) if packet.haslayer(Raw) else 0,
                    metadata={"http": http_analysis.__dict__}
                )
        
        # HTTPS-Analyse (nur Header-Informationen)
        elif port in [443, 8443]:
            return ProtocolAnalysis(
                protocol="HTTPS",
                source_mac=src_mac,
                dest_mac=dst_mac,
                source_ip=src_ip,
                dest_ip=dst_ip,
                port=port,
                payload_size=len(packet[Raw]) if packet.haslayer(Raw) else 0,
                metadata={"encrypted": True}
            )
        
        return None
    
    def _analyze_udp_packet(self, packet, src_mac: str, dst_mac: str,
                          src_ip: str, dst_ip: str) -> Optional[ProtocolAnalysis]:
        """Analysiert UDP-Pakete."""
        udp_layer = packet[UDP]
        port = udp_layer.dport
        
        # DNS-Analyse
        if port == 53:
            dns_analysis = self._analyze_dns(packet)
            if dns_analysis:
                return ProtocolAnalysis(
                    protocol="DNS",
                    source_mac=src_mac,
                    dest_mac=dst_mac,
                    source_ip=src_ip,
                    dest_ip=dst_ip,
                    port=port,
                    payload_size=len(packet[Raw]) if packet.haslayer(Raw) else 0,
                    metadata={"dns": dns_analysis.__dict__}
                )
        
        # DHCP-Analyse
        elif port in [67, 68]:
            dhcp_analysis = self._analyze_dhcp(packet)
            if dhcp_analysis:
                return ProtocolAnalysis(
                    protocol="DHCP",
                    source_mac=src_mac,
                    dest_mac=dst_mac,
                    source_ip=src_ip,
                    dest_ip=dst_ip,
                    port=port,
                    payload_size=len(packet[Raw]) if packet.haslayer(Raw) else 0,
                    metadata={"dhcp": dhcp_analysis.__dict__}
                )
        
        return None
    
    def _analyze_http(self, packet) -> Optional[HTTPAnalysis]:
        """Analysiert HTTP-Traffic."""
        try:
            if packet.haslayer(Raw):
                raw_data = bytes(packet[Raw])
                http_data = raw_data.decode('utf-8', errors='ignore')
                
                # HTTP-Request analysieren
                if b'HTTP/1.' in raw_data:
                    return self._parse_http_response(http_data)
                else:
                    return self._parse_http_request(http_data)
                    
        except Exception as e:
            logger.debug(f"HTTP-Analyse fehlgeschlagen: {e}")
            
        return None
    
    def _parse_http_request(self, http_data: str) -> Optional[HTTPAnalysis]:
        """Parst HTTP-Request."""
        lines = http_data.split('\n')
        if not lines:
            return None
            
        # Erste Zeile: Method, URL, Version
        first_line = lines[0].strip()
        parts = first_line.split()
        if len(parts) < 3:
            return None
            
        method = parts[0]
        url = parts[1]
        
        # Header analysieren
        headers = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
        
        host = headers.get('host', '')
        user_agent = headers.get('user-agent', '')
        
        # Verdächtige Patterns erkennen
        suspicious = []
        if any(pattern in url.lower() for pattern in ['admin', 'login', 'config']):
            suspicious.append('admin_access')
        if 'sql' in url.lower() or 'union' in url.lower():
            suspicious.append('sql_injection')
        if len(user_agent) > 200:
            suspicious.append('long_user_agent')
            
        return HTTPAnalysis(
            method=method,
            url=url,
            host=host,
            user_agent=user_agent,
            suspicious_patterns=suspicious
        )
    
    def _parse_http_response(self, http_data: str) -> Optional[HTTPAnalysis]:
        """Parst HTTP-Response."""
        lines = http_data.split('\n')
        if not lines:
            return None
            
        # Status-Line
        status_line = lines[0].strip()
        parts = status_line.split()
        if len(parts) < 2:
            return None
            
        status_code = int(parts[1]) if parts[1].isdigit() else None
        
        # Header analysieren
        headers = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
        
        content_type = headers.get('content-type', '')
        content_length = int(headers.get('content-length', 0)) if headers.get('content-length', '').isdigit() else None
        
        return HTTPAnalysis(
            method="RESPONSE",
            url="",
            host="",
            status_code=status_code,
            content_type=content_type,
            content_length=content_length
        )
    
    def _analyze_dns(self, packet) -> Optional[DNSAnalysis]:
        """Analysiert DNS-Traffic."""
        try:
            if packet.haslayer(DNS):
                dns_layer = packet[DNS]
                
                # Query analysieren
                if dns_layer.qr == 0:  # Query
                    if dns_layer.qd:
                        query = dns_layer.qd[0]
                        domain = query.qname.decode('utf-8').rstrip('.')
                        query_type = self._get_dns_type_name(query.qtype)
                        
                        # Verdächtige Domain prüfen
                        is_malicious = any(sus in domain.lower() for sus in self.suspicious_domains)
                        
                        return DNSAnalysis(
                            query_type=query_type,
                            domain=domain,
                            is_malicious=is_malicious
                        )
                
                # Response analysieren
                elif dns_layer.qr == 1:  # Response
                    answers = []
                    if dns_layer.an:
                        for rr in dns_layer.an:
                            if hasattr(rr, 'rdata'):
                                answers.append(str(rr.rdata))
                    
                    return DNSAnalysis(
                        query_type="RESPONSE",
                        domain="",
                        response_code=dns_layer.rcode,
                        answers=answers,
                        ttl=dns_layer.an[0].ttl if dns_layer.an else None
                    )
                    
        except Exception as e:
            logger.debug(f"DNS-Analyse fehlgeschlagen: {e}")
            
        return None
    
    def _analyze_dhcp(self, packet) -> Optional[DHCPAnalysis]:
        """Analysiert DHCP-Traffic."""
        try:
            if packet.haslayer(DHCP):
                dhcp_layer = packet[DHCP]
                bootp_layer = packet[BOOTP]
                
                # DHCP-Optionen extrahieren
                options = {}
                for option in dhcp_layer.options:
                    if isinstance(option, tuple) and len(option) == 2:
                        options[option[0]] = option[1]
                
                message_type = self._get_dhcp_message_type(options.get(53, 0))
                client_mac = bootp_layer.chaddr[:6].hex(':')
                requested_ip = str(bootp_layer.yiaddr) if bootp_layer.yiaddr else None
                server_ip = str(bootp_layer.siaddr) if bootp_layer.siaddr else None
                
                hostname = options.get(12, b'').decode('utf-8', errors='ignore') if options.get(12) else None
                vendor_class = options.get(60, b'').decode('utf-8', errors='ignore') if options.get(60) else None
                lease_time = options.get(51) if options.get(51) else None
                
                return DHCPAnalysis(
                    message_type=message_type,
                    client_mac=client_mac,
                    requested_ip=requested_ip,
                    server_ip=server_ip,
                    hostname=hostname,
                    vendor_class=vendor_class,
                    lease_time=lease_time
                )
                
        except Exception as e:
            logger.debug(f"DHCP-Analyse fehlgeschlagen: {e}")
            
        return None
    
    def _analyze_arp_packet(self, packet) -> Optional[ProtocolAnalysis]:
        """Analysiert ARP-Pakete."""
        try:
            if packet.haslayer(ARP):
                arp_layer = packet[ARP]
                return ProtocolAnalysis(
                    protocol="ARP",
                    source_mac=packet.src,
                    dest_mac=packet.dst,
                    source_ip=arp_layer.psrc,
                    dest_ip=arp_layer.pdst,
                    metadata={
                        "operation": "request" if arp_layer.op == 1 else "reply",
                        "hardware_type": arp_layer.hwtype,
                        "protocol_type": arp_layer.ptype
                    }
                )
        except Exception as e:
            logger.debug(f"ARP-Analyse fehlgeschlagen: {e}")
            
        return None
    
    def _get_dns_type_name(self, qtype: int) -> str:
        """Konvertiert DNS-Type-Code zu Name."""
        dns_types = {
            1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 12: "PTR",
            15: "MX", 16: "TXT", 28: "AAAA", 33: "SRV"
        }
        return dns_types.get(qtype, f"TYPE{qtype}")
    
    def _get_dhcp_message_type(self, msg_type: int) -> str:
        """Konvertiert DHCP-Message-Type zu Name."""
        dhcp_types = {
            1: "DISCOVER", 2: "OFFER", 3: "REQUEST", 
            4: "DECLINE", 5: "ACK", 6: "NAK", 7: "RELEASE", 8: "INFORM"
        }
        return dhcp_types.get(msg_type, f"UNKNOWN{msg_type}")
    
    def get_device_fingerprint(self, mac_address: str) -> Dict[str, Any]:
        """Erstellt Device-Fingerprint basierend auf DPI-Daten."""
        fingerprint = {
            "mac": mac_address,
            "protocols": set(),
            "dns_queries": [],
            "http_requests": [],
            "user_agents": set(),
            "suspicious_activity": []
        }
        
        # DNS-Queries sammeln
        if mac_address in self.dns_queries:
            fingerprint["dns_queries"] = self.dns_queries[mac_address]
            fingerprint["protocols"].add("DNS")
        
        # HTTP-Requests sammeln
        if mac_address in self.http_requests:
            fingerprint["http_requests"] = self.http_requests[mac_address]
            fingerprint["protocols"].add("HTTP")
            
            # User-Agents extrahieren
            for req in self.http_requests[mac_address]:
                if "user_agent" in req and req["user_agent"]:
                    fingerprint["user_agents"].add(req["user_agent"])
        
        # Verdächtige Aktivität prüfen
        for req in fingerprint["http_requests"]:
            if req.get("suspicious_patterns"):
                fingerprint["suspicious_activity"].extend(req["suspicious_patterns"])
        
        return fingerprint
    
    def get_network_insights(self) -> Dict[str, Any]:
        """Gibt Netzwerk-Insights basierend auf DPI-Daten zurück."""
        insights = {
            "total_dns_queries": sum(len(queries) for queries in self.dns_queries.values()),
            "total_http_requests": sum(len(requests) for requests in self.http_requests.values()),
            "unique_domains": set(),
            "suspicious_domains": set(),
            "device_types": defaultdict(int),
            "protocol_distribution": Counter()
        }
        
        # DNS-Insights
        for mac, queries in self.dns_queries.items():
            for query in queries:
                domain = query.get("domain", "")
                insights["unique_domains"].add(domain)
                if query.get("is_malicious", False):
                    insights["suspicious_domains"].add(domain)
        
        # HTTP-Insights
        for mac, requests in self.http_requests.items():
            for req in requests:
                insights["protocol_distribution"]["HTTP"] += 1
                
                # User-Agent basierte Geräte-Erkennung
                ua = req.get("user_agent", "").lower()
                if "android" in ua:
                    insights["device_types"]["Android"] += 1
                elif "iphone" in ua or "ipad" in ua:
                    insights["device_types"]["iOS"] += 1
                elif "windows" in ua:
                    insights["device_types"]["Windows"] += 1
                elif "linux" in ua:
                    insights["device_types"]["Linux"] += 1
        
        return insights
