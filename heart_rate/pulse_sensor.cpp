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
//   0.1uF ceramic capacitor across MCP3008 VDD <-> GND, as close to the chip as possible
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

// Simple moving-average low-pass filter to smooth out high-frequency noise
// (e.g. mains hum picked up when touching the sensor) while preserving the
// much slower heartbeat waveform (~1 Hz).
const int FILTER_WINDOW = 15; // ~30ms window at 2ms sampling
int filterBuf[FILTER_WINDOW] = {0};
int filterIndex = 0;
int filterCount = 0;
long filterSum = 0;

int smooth(int rawValue) {
    filterSum -= filterBuf[filterIndex];
    filterBuf[filterIndex] = rawValue;
    filterSum += rawValue;
    filterIndex = (filterIndex + 1) % FILTER_WINDOW;
    if (filterCount < FILTER_WINDOW) filterCount++;
    return (int)(filterSum / filterCount);
}

int main() {
    if (!spiInit()) {
        fprintf(stderr, "Failed to init SPI. Is SPI enabled (raspi-config) and did you reboot?\n");
        return 1;
    }

    // --- Peak detection state ---
    int   signalValue   = 0;
    int   threshold      = 550;   // adjust after watching raw/filtered values (see calibration line below)
    int   P               = 512;   // running peak (reset each beat cycle)
    int   T               = 512;   // running trough (reset each beat cycle)
    bool  pulseDetected  = false;
    auto  lastBeatTime    = std::chrono::steady_clock::now();
    int   sampleCount     = 0;

    // Refractory period: ignore any new beat within this many ms of the last one.
    // 250ms = max detectable 240 BPM, well above any real resting/exercise HR,
    // and long enough to reject double-triggers from noisy wiggles near threshold.
    const double REFRACTORY_MS = 250.0;

    // Rolling average of the last N beat intervals, so a single noisy interval
    // doesn't swing the displayed BPM wildly.
    const int AVG_WINDOW = 8;
    double ibiHistory[AVG_WINDOW];
    int    ibiCount = 0;   // how many valid intervals collected so far (caps at AVG_WINDOW)
    int    ibiIndex = 0;   // next slot to write

    printf("Starting pulse read. Place finger/earlobe on sensor, press firmly, avoid ambient light.\n");
    printf("Uncomment the calibration print line below if you need to check raw/filtered values.\n\n");

    while (true) {
        int raw = readADC(ADC_CHANNEL);
        if (raw < 0) break; // SPI error
        signalValue = smooth(raw);

        // printf("raw: %d  filtered: %d\n", raw, signalValue); // uncomment for calibration

        if (signalValue > P) P = signalValue;
        if (signalValue < T) T = signalValue;

        auto now = std::chrono::steady_clock::now();
        double sinceLastBeatMs = std::chrono::duration<double, std::milli>(now - lastBeatTime).count();

        if (signalValue > threshold && !pulseDetected && sinceLastBeatMs > REFRACTORY_MS) {
            pulseDetected = true;

            if (sinceLastBeatMs < 2000) { // upper bound only; refractory already enforces lower bound
                ibiHistory[ibiIndex] = sinceLastBeatMs;
                ibiIndex = (ibiIndex + 1) % AVG_WINDOW;
                if (ibiCount < AVG_WINDOW) ibiCount++;

                double sum = 0;
                for (int i = 0; i < ibiCount; i++) sum += ibiHistory[i];
                double avgIbi = sum / ibiCount;
                double bpm = 60000.0 / avgIbi;

                printf("Beat! Interval: %.0f ms  Instant BPM: %.1f  Avg(%d) BPM: %.1f\n",
                       sinceLastBeatMs, 60000.0 / sinceLastBeatMs, ibiCount, bpm);
            }

            lastBeatTime = now;
            // reset peak/trough tracking for the next beat cycle
            P = signalValue;
            T = signalValue;
        } else if (signalValue < threshold) {
            pulseDetected = false;
        }

        // slowly adapt threshold toward 60% between trough and peak seen so far
        if (++sampleCount % 100 == 0 && P > T) {
            threshold = T + (int)((P - T) * 0.6);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2)); // ~500 Hz sample rate
    }

    close(spiFd);
    return 0;
}