#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <unitree/robot/channel/channel_factory.hpp>

namespace unitree_cpp_detail {

inline std::uint32_t Crc32Core(const std::uint32_t* words, std::uint32_t length) noexcept {
    std::uint32_t crc = 0xFFFFFFFF;
    constexpr std::uint32_t polynomial = 0x04C11DB7;
    for (std::uint32_t index = 0; index < length; ++index) {
        std::uint32_t bit = 1U << 31;
        const std::uint32_t data = words[index];
        for (std::uint32_t offset = 0; offset < 32; ++offset) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc <<= 1;
            }
            if (data & bit) {
                crc ^= polynomial;
            }
            bit >>= 1;
        }
    }
    return crc;
}

struct DdsEndpoint {
    std::int32_t domain_id;
    std::string net_if;

    bool operator==(const DdsEndpoint& rhs) const noexcept {
        return domain_id == rhs.domain_id && net_if == rhs.net_if;
    }
};

class DdsEndpointInitGuard {
   public:
    template <class Initializer>
    bool InitializeOnce(std::int32_t domain_id, const std::string& net_if, Initializer&& initializer) {
        if (domain_id < 0) {
            throw std::invalid_argument("DDS domain_id must be non-negative");
        }
        if (net_if.empty()) {
            throw std::invalid_argument("DDS net_if must not be empty");
        }

        DdsEndpoint requested{domain_id, net_if};
        std::lock_guard<std::mutex> lock(mutex_);
        if (endpoint_) {
            if (*endpoint_ == requested) {
                return false;
            }
            throw std::runtime_error(
                "DDS ChannelFactory endpoint conflict: initialized domain " + std::to_string(endpoint_->domain_id) +
                " on '" + endpoint_->net_if + "', requested domain " + std::to_string(requested.domain_id) +
                " on '" + requested.net_if + "'");
        }

        std::forward<Initializer>(initializer)(requested.domain_id, requested.net_if);
        endpoint_.emplace(std::move(requested));
        return true;
    }

   private:
    std::mutex mutex_;
    std::optional<DdsEndpoint> endpoint_;
};

inline bool InitializeDdsEndpointOnce(std::int32_t domain_id, const std::string& net_if) {
    static DdsEndpointInitGuard guard;
    return guard.InitializeOnce(domain_id, net_if, [](std::int32_t selected_domain, const std::string& selected_if) {
        unitree::robot::ChannelFactory::Instance()->Init(selected_domain, selected_if);
    });
}

}  // namespace unitree_cpp_detail
