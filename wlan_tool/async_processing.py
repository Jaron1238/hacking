# Async/Concurrency Verbesserungen
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class PacketData:
    """Datenklasse für Packet-Informationen"""
    timestamp: float
    src_mac: str
    dst_mac: str
    data: bytes
    signal_strength: Optional[int] = None

class AsyncPacketProcessor:
    """Asynchroner Packet-Processor mit Queue-System"""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.packet_queue = asyncio.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_semaphore = asyncio.Semaphore(max_workers)
        self.is_running = False
        
    async def start_processing(self):
        """Startet asynchrone Packet-Verarbeitung"""
        self.is_running = True
        tasks = []
        
        # Starte mehrere Worker-Tasks
        for i in range(4):
            task = asyncio.create_task(self._packet_worker(f"worker-{i}"))
            tasks.append(task)
        
        logger.info(f"Gestartet {len(tasks)} Packet-Worker")
        return tasks
    
    async def _packet_worker(self, worker_name: str):
        """Worker-Task für Packet-Verarbeitung"""
        while self.is_running:
            try:
                # Warte auf Packet aus Queue
                packet = await asyncio.wait_for(
                    self.packet_queue.get(), timeout=1.0
                )
                
                async with self.processing_semaphore:
                    # Verarbeite Packet asynchron
                    await self._process_packet(packet, worker_name)
                    
                self.packet_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Fehler in {worker_name}: {e}")
    
    async def _process_packet(self, packet: PacketData, worker_name: str):
        """Verarbeitet einzelnes Packet"""
        # CPU-intensive Verarbeitung in Thread-Pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._analyze_packet, packet
        )
        
        logger.debug(f"{worker_name} verarbeitete Packet: {result}")
    
    def _analyze_packet(self, packet: PacketData) -> Dict[str, Any]:
        """CPU-intensive Packet-Analyse (läuft in Thread-Pool)"""
        # Simuliere komplexe Analyse
        time.sleep(0.01)  # Simuliere Verarbeitungszeit
        
        return {
            "src_vendor": self._lookup_vendor(packet.src_mac),
            "dst_vendor": self._lookup_vendor(packet.dst_mac),
            "packet_size": len(packet.data),
            "processed_at": time.time()
        }
    
    def _lookup_vendor(self, mac: str) -> str:
        """Vendor-Lookup (Placeholder)"""
        return "Unknown"
    
    async def add_packet(self, packet: PacketData):
        """Fügt Packet zur Verarbeitungsqueue hinzu"""
        try:
            await self.packet_queue.put(packet)
        except asyncio.QueueFull:
            logger.warning("Packet-Queue voll, Packet verworfen")
    
    async def stop_processing(self):
        """Stoppt Packet-Verarbeitung"""
        self.is_running = False
        await self.packet_queue.join()
        self.executor.shutdown(wait=True)

class AsyncOUIDownloader:
    """Asynchroner OUI-Downloader mit Connection-Pooling"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        # Connection-Pool für HTTP-Requests
        connector = aiohttp.TCPConnector(
            limit=10,  # Max 10 gleichzeitige Verbindungen
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "WLAN-Tool/2.1 (Async)",
                "Accept": "text/plain,text/html,*/*"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def download_oui_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, bytes]:
        """Lädt mehrere OUI-Quellen parallel herunter"""
        tasks = []
        
        for source in sources:
            task = asyncio.create_task(
                self._download_single_source(source)
            )
            tasks.append(task)
        
        # Warte auf alle Downloads (parallel)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sammle erfolgreiche Downloads
        successful_downloads = {}
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning(f"Download fehlgeschlagen für {source['name']}: {result}")
            else:
                successful_downloads[source['name']] = result
        
        return successful_downloads
    
    async def _download_single_source(self, source: Dict[str, Any]) -> bytes:
        """Lädt einzelne OUI-Quelle herunter"""
        async with self.session.get(source['url']) as response:
            if response.status == 200:
                data = await response.read()
                logger.info(f"✓ {source['name']}: {len(data):,} bytes")
                return data
            else:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )

class AsyncFileProcessor:
    """Asynchroner File-Processor für große PCAP-Dateien"""
    
    @staticmethod
    async def process_large_pcap(file_path: str, chunk_size: int = 8192) -> AsyncGenerator:
        """Streaming Processing für große PCAP-Dateien"""
        async with aiofiles.open(file_path, 'rb') as file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                # Verarbeite Chunk asynchron
                yield await AsyncFileProcessor._process_chunk(chunk)
    
    @staticmethod
    async def _process_chunk(chunk: bytes) -> Dict[str, Any]:
        """Verarbeitet einzelnen File-Chunk"""
        # Simuliere Chunk-Verarbeitung
        await asyncio.sleep(0.001)
        
        return {
            "chunk_size": len(chunk),
            "processed_at": time.time()
        }

# Beispiel-Usage
async def example_async_processing():
    """Beispiel für asynchrone Verarbeitung"""
    
    # Async Packet Processing
    processor = AsyncPacketProcessor(max_workers=4)
    tasks = await processor.start_processing()
    
    # Simuliere Packet-Eingabe
    for i in range(100):
        packet = PacketData(
            timestamp=time.time(),
            src_mac=f"00:11:22:33:44:{i:02x}",
            dst_mac="ff:ff:ff:ff:ff:ff",
            data=b"test_data"
        )
        await processor.add_packet(packet)
    
    # Warte kurz und stoppe
    await asyncio.sleep(2)
    await processor.stop_processing()
    
    # Async OUI Download
    async with AsyncOUIDownloader() as downloader:
        sources = [
            {"name": "Test", "url": "https://httpbin.org/bytes/1000"}
        ]
        results = await downloader.download_oui_sources(sources)
        logger.info(f"Downloads: {list(results.keys())}")

if __name__ == "__main__":
    asyncio.run(example_async_processing())