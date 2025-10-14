# data/led_controller.py

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Der Pfad zur Steuerung der roten Power-LED auf den meisten Raspberry Pi Modellen.
# Kann je nach Modell variieren (z.B. 'led1' für die grüne LED auf älteren Modellen).
LED_PATH = "/sys/class/leds/led1/"


class StatusLED(threading.Thread):
    """
    Steuert eine Status-LED, um die Aktivität des Skripts anzuzeigen.
    Lässt die rote Power-LED des Raspberry Pi blinken.
    """

    def __init__(self, blink_interval: float = 0.5):
        super().__init__(daemon=True, name="StatusLEDThread")
        self.blink_interval = blink_interval
        self.stop_event = threading.Event()
        self.original_trigger = "default-on"  # Standard-Verhalten der PWR-LED

    def _set_led_trigger(self, trigger: str):
        """Setzt den Trigger für die LED (z.B. 'none', 'timer', 'default-on')."""
        try:
            with open(LED_PATH + "trigger", "w") as f:
                f.write(trigger)
            return True
        except (IOError, FileNotFoundError) as e:
            logger.debug(
                f"Konnte LED-Trigger nicht setzen. Pfad {LED_PATH} möglicherweise falsch? Fehler: {e}"
            )
            return False

    def _set_led_brightness(self, value: int):
        """Setzt die Helligkeit der LED (0 = aus, 1 = an)."""
        try:
            with open(LED_PATH + "brightness", "w") as f:
                f.write(str(value))
        except IOError:
            pass  # Kann fehlschlagen, wenn Trigger nicht auf 'none' steht

    def run(self):
        """Die Hauptschleife, die die LED blinken lässt."""
        # Speichere den ursprünglichen Trigger, um ihn später wiederherzustellen
        try:
            with open(LED_PATH + "trigger", "r") as f:
                self.original_trigger = f.read().strip()
        except IOError:
            logger.warning("Konnte ursprünglichen LED-Trigger nicht lesen.")

        # Übernehme die Kontrolle über die LED
        if not self._set_led_trigger("none"):
            logger.error(
                "Keine Kontrolle über die LED möglich. Status-LED wird nicht funktionieren."
            )
            return

        logger.info("Status-LED aktiviert. Die rote PWR-LED blinkt jetzt.")

        while not self.stop_event.is_set():
            self._set_led_brightness(1)  # LED an
            time.sleep(self.blink_interval)
            self._set_led_brightness(0)  # LED aus
            time.sleep(self.blink_interval)

    def stop(self):
        """Stoppt das Blinken und stellt den Originalzustand der LED wieder her."""
        self.stop_event.set()
        # Warte kurz, damit der Thread die Schleife beenden kann
        self.join(timeout=self.blink_interval * 2)

        # Stelle den ursprünglichen Trigger wieder her (z.B. 'default-on')
        logger.info("Stelle ursprüngliches Verhalten der Status-LED wieder her.")
        self._set_led_trigger(self.original_trigger)
        # Helligkeit auf 1 setzen, falls der Trigger 'none' war
        self._set_led_brightness(1)
