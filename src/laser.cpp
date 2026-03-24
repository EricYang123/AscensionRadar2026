#include "laser.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

laser::laser(const char* device, int baudrate) {
    fd = -1;
    init(device, baudrate);
}

laser::~laser() {
    closePort();
}

bool laser::init(const char* device, int baudrate) {
    fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "Error opening serial port\n";
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof tty);

    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error from tcgetattr\n";
        closePort();
        return false;
    }

    // Set baud rate
    cfsetospeed(&tty, baudrate);
    cfsetispeed(&tty, baudrate);

    // 8N1 mode
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~PARENB;   // No parity
    tty.c_cflag &= ~CSTOPB;   // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;  // No flow control
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 10; // 1 second read timeout

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error from tcsetattr\n";
        closePort();
        return false;
    }

    return true;
}

bool laser::send(const uint8_t* data, size_t len) {
    if (fd < 0) return false;

    ssize_t written = write(fd, data, len);
    return written == (ssize_t)len;
}

void laser::closePort() {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

void laser::setServoAngle(int servox, int servoy){

        uint8_t buffer[8];
        buffer[0] = static_cast<uint8_t> (servox & 0xFF);
        buffer[1] = static_cast<uint8_t> ((servox >> 8) & 0xFF);
        buffer[2] = static_cast<uint8_t> ((servox >> 16) & 0xFF);
        buffer[3] = static_cast<uint8_t> ((servox >> 24) & 0xFF);

        buffer[4] = static_cast<uint8_t> (servoy & 0xFF);
        buffer[5] = static_cast<uint8_t> ((servoy >> 8) & 0xFF);
        buffer[6] = static_cast<uint8_t> ((servoy >> 16) & 0xFF);
        buffer[7] = static_cast<uint8_t> ((servoy >> 24) & 0xFF);
        // cout << message << "\n";
        send( buffer, sizeof(buffer));
}