#ifndef FOMS_LOG_HPP
#define FOMS_LOG_HPP

#include <iostream>
/**
 * @file
 * @brief Defines log class.
 */

namespace foms {

/// @brief verbosity level for logs
enum class LogVerbosity { ALL = 0, ERROR_ONLY, NOLOG };

/// @brief error level
enum class errorLevel { INFO = 0, ERROR };

/// @brief  class for logging to stream with different levels of verbosity
class LogStream {

  public:
    /**
     * @brief Default constructor that initializes the LogStream to use std::cout.
     */
    LogStream() : m_stream(&std::cout){};

    /**
     * @brief Constructor that initializes the LogStream to use a specified output stream.
     * @param out_stream The output stream to be used by the LogStream.
     */
    explicit LogStream(std::ostream& out_stream) : m_stream(&out_stream){};

    /**
     * @brief Writes a log message to the output stream if the error level is greater than or equal to the
     * current verbosity level.
     * @param log The log message to be written.
     * @param level The error level of the log message.
     */
    void writeLog(const std::string& log, errorLevel level) {

        if (static_cast< int >(level) >= static_cast< int >(m_verbosity)) {
            *m_stream << log << '\n';
        }
    };

    /**
     * @brief Sets the verbosity level of the LogStream.
     * @param verbosity The verbosity level to be set.
     */
    void set_verbosity(LogVerbosity verbosity) { m_verbosity = verbosity; };

  private:
    std::ostream* m_stream;                             ///< The output stream used by the LogStream.
    LogVerbosity m_verbosity = foms::LogVerbosity::ALL; ///< The current verbosity level of the LogStream.
};

static LogStream g_logger; // NOLINT //NOSONAR

} // namespace foms

#endif // FMS_LOG_HPP