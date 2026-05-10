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
    void aimLaser(int detectx, int detecty);
    void closePort();

private:
    int fd;
    int currentServoX = 135;
    int currentServoY = 135;
};

#endif // SERIALPORT_H
