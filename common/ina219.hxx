#ifndef INA219_HXX
#define INA219_HXX

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

struct INA219Reading {
    double bus_voltage_v;
    double shunt_voltage_mv;
    double current_ma;
    double power_mw;
};

struct INA219Stats {
    double bus_voltage_v_avg;
    double bus_voltage_v_min;
    double bus_voltage_v_max;
    double shunt_voltage_mv_avg;
    double current_ma_avg;
    double current_ma_min;
    double current_ma_max;
    double power_mw_avg;
    double power_mw_min;
    double power_mw_max;
    double energy_mj;
    int sample_count;
};

class INA219 {
   public:
    static constexpr uint8_t INA219_ADDR = 0x40;
    static constexpr uint8_t REG_CONFIG = 0x00;
    static constexpr uint8_t REG_SHUNT_V = 0x01;
    static constexpr uint8_t REG_BUS_V = 0x02;
    static constexpr uint8_t REG_POWER = 0x03;
    static constexpr uint8_t REG_CURRENT = 0x04;
    static constexpr uint8_t REG_CALIBRATION = 0x05;
    static constexpr uint16_t CAL_VALUE = 4096;
    static constexpr double CURRENT_LSB_MA = 0.1;
    static constexpr double POWER_LSB_MW = CURRENT_LSB_MA * 20.0;

    INA219() : i2c_fd_(-1) {}

    ~INA219() { close_device(); }

    bool open_device(const char* dev = "/dev/i2c-1") {
#ifdef __linux__
        i2c_fd_ = ::open(dev, O_RDWR);
        if (i2c_fd_ < 0) {
            return false;
        }
        if (ioctl(i2c_fd_, I2C_SLAVE, INA219_ADDR) < 0) {
            close_device();
            return false;
        }
        return true;
#else
        (void) dev;
        return false;
#endif
    }

    bool configure() {
#ifdef __linux__
        if (i2c_fd_ < 0) {
            return false;
        }
        if (!write_reg16(REG_CONFIG, 0x8000)) {
            return false;
        }
        usleep(10000);
        if (!write_reg16(REG_CONFIG, 0x399F)) {
            return false;
        }
        return write_reg16(REG_CALIBRATION, CAL_VALUE);
#else
        return false;
#endif
    }

    bool read_reading(INA219Reading& reading) {
#ifdef __linux__
        int16_t raw_bus, raw_shunt, raw_current, raw_power;
        if (!read_reg16(REG_BUS_V, raw_bus)) {
            return false;
        }
        if (!read_reg16(REG_SHUNT_V, raw_shunt)) {
            return false;
        }
        if (!read_reg16(REG_CURRENT, raw_current)) {
            return false;
        }
        if (!read_reg16(REG_POWER, raw_power)) {
            return false;
        }

        reading.bus_voltage_v = ((raw_bus >> 3) * 4) / 1000.0;
        reading.shunt_voltage_mv = raw_shunt * 0.01;
        reading.current_ma = raw_current * CURRENT_LSB_MA;
        reading.power_mw = raw_power * POWER_LSB_MW;
        return true;
#else
        (void) reading;
        return false;
#endif
    }

    void close_device() {
#ifdef __linux__
        if (i2c_fd_ >= 0) {
            ::close(i2c_fd_);
            i2c_fd_ = -1;
        }
#endif
    }

    bool is_open() const { return i2c_fd_ >= 0; }

   private:
    int i2c_fd_;

#ifdef __linux__
    bool write_reg16(uint8_t reg, uint16_t value) {
        uint8_t buf[3];
        buf[0] = reg;
        buf[1] = (value >> 8) & 0xFF;
        buf[2] = value & 0xFF;
        return ::write(i2c_fd_, buf, 3) == 3;
    }

    bool read_reg16(uint8_t reg, int16_t& value) {
        if (::write(i2c_fd_, &reg, 1) != 1) {
            return false;
        }
        uint8_t buf[2];
        if (::read(i2c_fd_, buf, 2) != 2) {
            return false;
        }
        value = (int16_t) ((buf[0] << 8) | buf[1]);
        return true;
    }
#endif
};

class INA219Sampler {
   public:
    INA219Sampler() : running_(false), sample_interval_us_(100000) {}

    void set_sample_interval_us(int interval_us) { sample_interval_us_ = interval_us; }

    bool start(INA219* sensor) {
        if (sensor == nullptr || !sensor->is_open() || running_) {
            return false;
        }
        sensor_ = sensor;
        readings_.clear();
        running_ = true;
        thread_ = std::thread(&INA219Sampler::sample_loop, this);
        return true;
    }

    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    INA219Stats get_stats() const {
        std::lock_guard<std::mutex> lock(readings_mutex_);
        INA219Stats stats = {};
        stats.sample_count = (int) readings_.size();
        if (readings_.empty()) {
            return stats;
        }

        stats.bus_voltage_v_min = stats.bus_voltage_v_max = readings_[0].bus_voltage_v;
        stats.current_ma_min = stats.current_ma_max = readings_[0].current_ma;
        stats.power_mw_min = stats.power_mw_max = readings_[0].power_mw;

        double bus_v_sum = 0.0;
        double shunt_mv_sum = 0.0;
        double current_ma_sum = 0.0;
        double power_mw_sum = 0.0;

        for (const INA219Reading& r : readings_) {
            bus_v_sum += r.bus_voltage_v;
            shunt_mv_sum += r.shunt_voltage_mv;
            current_ma_sum += r.current_ma;
            power_mw_sum += r.power_mw;

            if (r.bus_voltage_v < stats.bus_voltage_v_min) {
                stats.bus_voltage_v_min = r.bus_voltage_v;
            }
            if (r.bus_voltage_v > stats.bus_voltage_v_max) {
                stats.bus_voltage_v_max = r.bus_voltage_v;
            }
            if (r.current_ma < stats.current_ma_min) {
                stats.current_ma_min = r.current_ma;
            }
            if (r.current_ma > stats.current_ma_max) {
                stats.current_ma_max = r.current_ma;
            }
            if (r.power_mw < stats.power_mw_min) {
                stats.power_mw_min = r.power_mw;
            }
            if (r.power_mw > stats.power_mw_max) {
                stats.power_mw_max = r.power_mw;
            }
        }

        int n = stats.sample_count;
        stats.bus_voltage_v_avg = bus_v_sum / n;
        stats.shunt_voltage_mv_avg = shunt_mv_sum / n;
        stats.current_ma_avg = current_ma_sum / n;
        stats.power_mw_avg = power_mw_sum / n;

        // Energy (mJ) = sum(power_mw * dt_ms) / 1000
        double dt_ms = sample_interval_us_ / 1000.0;
        stats.energy_mj = (power_mw_sum * dt_ms) / 1000.0;

        return stats;
    }

   private:
    INA219* sensor_;
    std::atomic<bool> running_;
    int sample_interval_us_;
    std::thread thread_;
    mutable std::mutex readings_mutex_;
    std::vector<INA219Reading> readings_;

    void sample_loop() {
        while (running_) {
            INA219Reading reading;
            if (sensor_->read_reading(reading)) {
                std::lock_guard<std::mutex> lock(readings_mutex_);
                readings_.push_back(reading);
            }
#ifdef __linux__
            usleep(sample_interval_us_);
#endif
        }
    }
};

#endif
