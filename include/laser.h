#ifndef SERIALPORT_H
#define SERIALPORT_H

#include <cstddef>
#include <cstdint>
#include <termios.h>

class laser {
public:
    laser(const char* device, int baudrate);
    ~laser();

    bool init(const char* device, int baudrate);
    bool send(const uint8_t* data, size_t len);
    void setServoAngle(int servox, int servoy);
    void closePort();

private:
    int fd;
};

#endif // SERIALPORT_H
