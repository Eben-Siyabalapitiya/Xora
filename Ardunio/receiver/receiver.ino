// ============================================================
//  XORA — Receiver Firmware
//  Board: ESP32-WROOM-32 (Aideepen 30PIN)
//  Role:  Boards B, C, D — reads CSI data and sends to laptop
//
//  IMPORTANT: Before flashing each board, change BOARD_ID:
//    Board B → "B"
//    Board C → "C"
//    Board D → "D"

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

// ---- CHANGE THESE ----
const char* SSID      = "BELL991";      // same WiFi as transmitter
const char* PASSWORD  = "531D1374E947";  // same WiFi as transmitter
const char* LAPTOP_IP = "192.168.2.32";      // same IP as transmitter
const int   UDP_PORT  = 5005;
const char* BOARD_ID  = "B";                 // ← change to "C" or "D" for other boards
// ----------------------

WiFiUDP udp;
volatile int csiPacketCount = 0;

// ============================================================
//  CSI CALLBACK
//  This fires automatically every time the ESP32 receives
//  a WiFi packet and captures the channel state info.
//  We convert I/Q pairs to amplitude and send via UDP.
// ============================================================
void IRAM_ATTR csi_callback(void* ctx, wifi_csi_info_t* data) {
  if (!data || !data->buf || data->len < 2) return;

  // Start building the UDP packet string
  // Format: XORA_CSI,<board>,<timestamp>,<amp1>,<amp2>,...
  String packet = "XORA_CSI,";
  packet += BOARD_ID;
  packet += ",";
  packet += String(millis());

  int8_t* buf = data->buf;
  int     len = data->len;

  // Each subcarrier is 2 bytes: imaginary (Q) then real (I)
  // Amplitude = sqrt(I^2 + Q^2)
  for (int i = 0; i < len - 1; i += 2) {
    float I   = (float)buf[i + 1];
    float Q   = (float)buf[i];
    float amp = sqrtf(I * I + Q * Q);
    packet += ",";
    packet += String((int)amp);
  }

  // Send to laptop backend
  udp.beginPacket(LAPTOP_IP, UDP_PORT);
  udp.print(packet);
  udp.endPacket();

  csiPacketCount++;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n=== XORA Receiver (Board " + String(BOARD_ID) + ") ===");

  // Must be STA mode for CSI to work
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
    while (true) delay(1000);
  }

  Serial.println("\nConnected!");
  Serial.print("Board ");
  Serial.print(BOARD_ID);
  Serial.print(" IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);

  // ---- Enable CSI collection ----
  esp_wifi_set_promiscuous(true);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en           = true;   // legacy long training field
  cfg.htltf_en          = true;   // HT long training field
  cfg.stbc_htltf2_en    = true;   // STBC HT long training field
  cfg.ltf_merge_en      = true;   // merge LTFs for better accuracy
  cfg.channel_filter_en = false;  // raw channel data
  cfg.manu_scale        = false;
  cfg.shift             = false;

  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(&csi_callback, NULL);
  esp_wifi_set_csi(true);

  Serial.println("CSI collection active — sending to " + String(LAPTOP_IP));
}

void loop() {
  // Print heartbeat every 5 seconds so you can see it's alive
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 5000) {
    Serial.print("Board ");
    Serial.print(BOARD_ID);
    Serial.print(" — CSI packets sent: ");
    Serial.println(csiPacketCount);
    lastPrint = millis();
  }

  delay(10);
}
