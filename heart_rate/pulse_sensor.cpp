// Pulse Sensor reader for Raspberry Pi Zero
// Hardware: Pulse Sensor -> MCP3008 (SPI ADC) -> Pi Zero
// Uses the Linux spidev kernel driver directly (no wiringPi needed —
// wiringPi is deprecated/unavailable on current Raspberry Pi OS).
//
// Wiring:
//   Pulse Sensor S   -> MCP3008 CH0
//   Pulse Sensor +   -> 3.3V (Pi pin 1)
//   Pulse Sensor -   -> MCP3008 AGND
//   MCP3008 VDD      -> 3.3V (Pi pin 1)
//   MCP3008 VREF     -> 3.3V (Pi pin 1)
//   MCP3008 AGND     -> GND (Pi pin 6)
//   MCP3008 DGND     -> GND (Pi pin 6)
//   MCP3008 CLK      -> Pi SPI0 SCLK  (GPIO11 / pin23)
//   MCP3008 DOUT     -> Pi SPI0 MISO  (GPIO9  / pin21)
//   MCP3008 DIN      -> Pi SPI0 MOSI  (GPIO10 / pin19)
//   MCP3008 CS       -> Pi SPI0 CE0   (GPIO8  / pin24)
//
// Setup:
//   sudo raspi-config -> Interface Options -> SPI -> Enable -> reboot
//   (spidev needs no extra apt package; /dev/spidev0.0 appears after enabling+reboot)
//
// Build:
//   g++ -o pulse_sensor pulse_sensor.cpp
// Run:
//   sudo ./pulse_sensor
//   (or add your user to the 'spi' group to avoid needing sudo)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

static const char* SPI_DEVICE = "/dev/spidev0.0";
static const uint32_t SPI_SPEED = 1000000; // 1 MHz
static const uint8_t  SPI_BITS  = 8;
static const uint8_t  SPI_MODE  = SPI_MODE_0;
static const int      ADC_CHANNEL = 0; // MCP3008 CH0

int spiFd = -1;

bool spiInit() {
    spiFd = open(SPI_DEVICE, O_RDWR);
    if (spiFd < 0) {
        perror("open spidev");
        return false;
    }
    if (ioctl(spiFd, SPI_IOC_WR_MODE, &SPI_MODE) < 0) { perror("SPI_IOC_WR_MODE"); return false; }
    if (ioctl(spiFd, SPI_IOC_WR_BITS_PER_WORD, &SPI_BITS) < 0) { perror("SPI_IOC_WR_BITS_PER_WORD"); return false; }
    if (ioctl(spiFd, SPI_IOC_WR_MAX_SPEED_HZ, &SPI_SPEED) < 0) { perror("SPI_IOC_WR_MAX_SPEED_HZ"); return false; }
    return true;
}

// Read a single-ended channel (0-7) from the MCP3008, returns 0-1023
int readADC(int channel) {
    uint8_t tx[3] = {
        0x01,
        (uint8_t)((0x08 | channel) << 4),
        0x00
    };
    uint8_t rx[3] = {0, 0, 0};

    struct spi_ioc_transfer tr;
    memset(&tr, 0, sizeof(tr));
    tr.tx_buf = (unsigned long)tx;
    tr.rx_buf = (unsigned long)rx;
    tr.len = 3;
    tr.speed_hz = SPI_SPEED;
    tr.bits_per_word = SPI_BITS;

    if (ioctl(spiFd, SPI_IOC_MESSAGE(1), &tr) < 0) {
        perror("SPI_IOC_MESSAGE");
        return -1;
    }

    return ((rx[1] & 0x03) << 8) | rx[2];
}

int main() {
    if (!spiInit()) {
        fprintf(stderr, "Failed to init SPI. Is SPI enabled (raspi-config) and did you reboot?\n");
        return 1;
    }

    // --- Peak detection state, adapted from the PulseSensor Amped Arduino algorithm ---
    int   signalValue   = 0;
    int   threshold      = 550;   // adjust after watching raw values printed below
    int   P               = 512;   // running peak
    int   T               = 512;   // running trough
    bool  pulseDetected  = false;
    auto  lastBeatTime    = std::chrono::steady_clock::now();
    double bpm            = 0.0;
    int   sampleCount     = 0;

    printf("Starting pulse read. Place finger/earlobe on sensor.\n");
    printf("Uncomment the raw print line below to calibrate 'threshold' first.\n\n");

    while (true) {
        signalValue = readADC(ADC_CHANNEL);
        if (signalValue < 0) break; // SPI error

        if (signalValue > P) P = signalValue;
        if (signalValue < T) T = signalValue;

        if (signalValue > threshold && !pulseDetected) {
            pulseDetected = true;
            auto now = std::chrono::steady_clock::now();
            double intervalMs = std::chrono::duration<double, std::milli>(now - lastBeatTime).count();
            lastBeatTime = now;

            if (intervalMs > 300 && intervalMs < 2000) { // reject noise: 30-200 BPM range
                bpm = 60000.0 / intervalMs;
                printf("Beat! Interval: %.0f ms  BPM: %.1f\n", intervalMs, bpm);
            }
        } else if (signalValue < threshold) {
            pulseDetected = false;
        }

        if (++sampleCount % 500 == 0) {
            P -= (P - T) / 4;
            T += (P - T) / 4;
            threshold = T + (P - T) * 0.6;
        }

        // printf("raw: %d\n", signalValue); // uncomment for calibration

        std::this_thread::sleep_for(std::chrono::milliseconds(2)); // ~500 Hz sample rate
    }

    close(spiFd);
    return 0;
}