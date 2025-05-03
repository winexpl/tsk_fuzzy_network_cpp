#include <string>
#include <fstream>
#include <mutex>

class Logger
{
    public:
        static Logger& getInstance();
    
        void log(const std::string& message);
        void logInfo(const std::string& message);
        void logError(const std::string& message);
        void logDebug(const std::string& message);

    
    private:
        Logger();
        ~Logger();
        std::ofstream logFile_;
        std::mutex mutex_;
    };
    