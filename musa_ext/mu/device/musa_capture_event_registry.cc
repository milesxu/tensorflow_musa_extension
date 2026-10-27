#include "musa_capture_event_registry.h"

#include <mutex>
#include <unordered_map>

namespace tensorflow {
namespace musa {
namespace {

std::mutex& CaptureEventMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<unsigned long long, std::vector<musaEvent_t>>&
CaptureEvents() {
  static std::unordered_map<unsigned long long, std::vector<musaEvent_t>>
      events;
  return events;
}

}  // namespace

void RetainCaptureDependencyEvent(musaEvent_t event,
                                  unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  CaptureEvents()[capture_id].push_back(event);
}

std::vector<musaEvent_t> TakeCaptureDependencyEvents(
    unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  auto& events = CaptureEvents();
  auto it = events.find(capture_id);
  if (it == events.end()) {
    return {};
  }
  std::vector<musaEvent_t> taken = std::move(it->second);
  events.erase(it);
  return taken;
}

std::size_t PendingCaptureDependencyEventCount(
    unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  auto& events = CaptureEvents();
  auto it = events.find(capture_id);
  return it == events.end() ? 0 : it->second.size();
}

}  // namespace musa
}  // namespace tensorflow
