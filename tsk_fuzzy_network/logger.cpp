#include <mutex>
#include <string>
#include <iostream>
#include <fstream>
#include "logger.h"
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

Logger& Logger::getInstance()
{
    static Logger instance; // Singleton
    return instance;
}
    
void Logger::log(const std::string& message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    logFile_ << message << std::endl;
}

void Logger::logInfo(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    BOOST_LOG_TRIVIAL(info) << message;
}

void Logger::logError(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    BOOST_LOG_TRIVIAL(error) << message;
}

void Logger::logDebug(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    BOOST_LOG_TRIVIAL(debug) << message;
}

Logger::Logger()
{
    // boost::log::add_file_log("tsk.log");
    boost::log::add_console_log(std::cout, boost::log::keywords::format = "%Time% [%Severity%]: %Message%");
    // boost::log::add_common_attributes();
}

Logger::~Logger()
{

}