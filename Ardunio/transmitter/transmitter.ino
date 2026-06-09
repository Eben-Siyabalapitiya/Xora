// ============================================================
//  XORA — Transmitter Firmware
//  Board: LoR Core v3 (ESP32-WROOM-32)
//  Role:  Board A — sends WiFi packets constantly so the
//         3 receiver boards can measure CSI changes
// ============================================================

#include <WiFi.h>
#include <WiFiUdp.h>

// ---- CHANGE THESE ----
const char* SSID       = BELL991;       // your WiFi name
const char* PASSWORD   = 531D1374E947;   // your WiFi password
const char* LAPTOP_IP  = 192.168.2.32;       // run ipconfig and paste your IPv4 here
const int   UDP_PORT   = 5005;
// ----------------------

WiFiUDP udp;
int     packetCount = 0;

// LoR Core v3 has 4 onboard WS2812 LEDs on GPIO 33
// We use the built-in LED to show status (no extra library needed for basic blink)
#define STATUS_LED 2   // fallback GPIO if LED_DATA (33) conflicts

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== XORA Transmitter (Board A) ===");

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  Serial.print("Connecting to WiFi");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nFailed to connect. Check SSID/PASSWORD and restart.");
    // Blink fast to show error
    pinMode(STATUS_LED, OUTPUT);
    while (true) {
      digitalWrite(STATUS_LED, HIGH); delay(100);
      digitalWrite(STATUS_LED, LOW);  delay(100);
    }
  }

  Serial.println("\nConnected!");
  Serial.print("Board A IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Sending to: ");
  Serial.print(LAPTOP_IP);
  Serial.print(":");
  Serial.println(UDP_PORT);

  udp.begin(UDP_PORT);

  Serial.println("Transmitting at 100Hz...");
}

void loop() {
  // Build packet: XORA_TX , packet number , timestamp ms
  // The receivers on B/C/D will see this signal and measure
  // how it changes as a person moves through the room
  String msg = "XORA_TX," + String(packetCount) + "," + String(millis());

  udp.beginPacket(LAPTOP_IP, UDP_PORT);
  udp.print(msg);
  udp.endPacket();

  packetCount++;

  // Every 500 packets (~5 seconds) print a heartbeat
  if (packetCount % 500 == 0) {
    Serial.print("Sent ");
    Serial.print(packetCount);
    Serial.print(" packets | uptime: ");
    Serial.print(millis() / 1000);
    Serial.println("s");
  }

  delay(10); // 100Hz — 10ms between packets
}
