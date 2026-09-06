#ifndef EXAMM_INA219_HXX
#define EXAMM_INA219_HXX

// Minimal INA219 power monitor reader over Linux i2c-dev, plus a background
// sampler that integrates energy while inference runs. Header-only so it
// needs no CMake changes. On non-Linux systems open_device() always fails,
// so callers fall back to running without power monitoring.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#ifdef __linux__
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

struct INA219Stats {
    int32_t sample_count = 0;
    double bus_voltage_v_avg = 0, bus_voltage_v_min = 0, bus_voltage_v_max = 0;
    double shunt_voltage_mv_avg = 0;
    double current_ma_avg = 0, current_ma_min = 0, current_ma_max = 0;
    double power_mw_avg = 0, power_mw_min = 0, power_mw_max = 0;
    double energy_mj = 0;
};

class INA219 {
   private:
    int fd = -1;
    // Adafruit 32V/2A calibration: current LSB 0.1 mA, power LSB 2 mW
    static const uint16_t CALIBRATION = 4096;

    bool write_register(uint8_t reg, uint16_t value) {
#ifdef __linux__
        uint8_t buf[3] = {reg, (uint8_t) (value >> 8), (uint8_t) (value & 0xFF)};
        return write(fd, buf, 3) == 3;
#else
        return false;
#endif
    }

    bool read_register(uint8_t reg, uint16_t& value) {
#ifdef __linux__
        uint8_t buf[2];
        if (write(fd, &reg, 1) != 1 || read(fd, buf, 2) != 2) {
            return false;
        }
        value = (buf[0] << 8) | buf[1];
        return true;
#else
        return false;
#endif
    }

   public:
    bool open_device(const char* device, int address = 0x40) {
#ifdef __linux__
        fd = open(device, O_RDWR);
        if (fd < 0 || ioctl(fd, I2C_SLAVE, address) < 0) {
            close_device();
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    bool configure() {
        // 32V range, gain /8 (320mV), 12-bit bus + shunt ADC, continuous mode
        return write_register(0x05, CALIBRATION) && write_register(0x00, 0x399F);
    }

    void close_device() {
#ifdef __linux__
        if (fd >= 0) {
            close(fd);
        }
#endif
        fd = -1;
    }

    // all readings default to 0 if the read fails
    double bus_voltage_v() {
        uint16_t raw = 0;
        read_register(0x02, raw);
        return (raw >> 3) * 0.004;
    }
    double shunt_voltage_mv() {
        uint16_t raw = 0;
        read_register(0x01, raw);
        return (int16_t) raw * 0.01;
    }
    double current_ma() {
        uint16_t raw = 0;
        read_register(0x04, raw);
        return (int16_t) raw * 0.1;
    }
    double power_mw() {
        uint16_t raw = 0;
        read_register(0x03, raw);
        return raw * 2.0;
    }
};

class INA219Sampler {
   private:
    std::atomic<bool> running{false};
    std::thread sampler;
    INA219Stats stats;
    double bus_sum = 0, shunt_sum = 0, current_sum = 0, power_sum = 0;

    void run(INA219* ina219) {
        auto last = std::chrono::steady_clock::now();
        while (running) {
            double bus = ina219->bus_voltage_v(), shunt = ina219->shunt_voltage_mv();
            double current = ina219->current_ma(), power = ina219->power_mw();
            auto now = std::chrono::steady_clock::now();
            double dt_s = std::chrono::duration<double>(now - last).count();
            last = now;

            if (stats.sample_count == 0) {
                stats.bus_voltage_v_min = stats.bus_voltage_v_max = bus;
                stats.current_ma_min = stats.current_ma_max = current;
                stats.power_mw_min = stats.power_mw_max = power;
            }
            stats.bus_voltage_v_min = std::min(stats.bus_voltage_v_min, bus);
            stats.bus_voltage_v_max = std::max(stats.bus_voltage_v_max, bus);
            stats.current_ma_min = std::min(stats.current_ma_min, current);
            stats.current_ma_max = std::max(stats.current_ma_max, current);
            stats.power_mw_min = std::min(stats.power_mw_min, power);
            stats.power_mw_max = std::max(stats.power_mw_max, power);
            bus_sum += bus, shunt_sum += shunt, current_sum += current, power_sum += power;
            stats.energy_mj += power * dt_s;
            stats.sample_count++;

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

   public:
    void start(INA219* ina219) {
        stats = INA219Stats();
        bus_sum = shunt_sum = current_sum = power_sum = 0;
        running = true;
        sampler = std::thread(&INA219Sampler::run, this, ina219);
    }

    void stop() {
        running = false;
        if (sampler.joinable()) {
            sampler.join();
        }
        if (stats.sample_count > 0) {
            stats.bus_voltage_v_avg = bus_sum / stats.sample_count;
            stats.shunt_voltage_mv_avg = shunt_sum / stats.sample_count;
            stats.current_ma_avg = current_sum / stats.sample_count;
            stats.power_mw_avg = power_sum / stats.sample_count;
        }
    }

    INA219Stats get_stats() const {
        return stats;
    }
};

#endif
