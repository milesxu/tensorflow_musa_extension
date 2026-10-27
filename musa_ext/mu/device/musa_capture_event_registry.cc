#include "musa_capture_event_registry.h"

#include <map>
#include <mutex>
#include <utility>

namespace tensorflow {
namespace musa {
namespace {

std::mutex& CaptureEventMutex() {
  static std::mutex mutex;
  return mutex;
}

struct CaptureEventGroup {
  bool sealed = false;
  std::vector<musaEvent_t> events;
};

using CaptureEventKey = std::pair<int, unsigned long long>;

std::map<CaptureEventKey, CaptureEventGroup>& CaptureEvents() {
  static std::map<CaptureEventKey, CaptureEventGroup> events;
  return events;
}

}  // namespace

bool RetainCaptureDependencyEvent(musaEvent_t event, int device_id,
                                  unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  CaptureEventGroup& group =
      CaptureEvents()[CaptureEventKey(device_id, capture_id)];
  if (group.sealed) {
    return false;
  }
  group.events.push_back(event);
  return true;
}

std::vector<musaEvent_t> TakeCaptureDependencyEvents(
    int device_id, unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  auto& events = CaptureEvents();
  auto it = events.find(CaptureEventKey(device_id, capture_id));
  if (it == events.end()) {
    return {};
  }
  it->second.sealed = true;
  std::vector<musaEvent_t> taken = std::move(it->second.events);
  it->second.events.clear();
  return taken;
}

std::size_t PendingCaptureDependencyEventCount(
    int device_id, unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  auto& events = CaptureEvents();
  auto it = events.find(CaptureEventKey(device_id, capture_id));
  return it == events.end() ? 0 : it->second.events.size();
}

void ReleaseCaptureDependencyGate(int device_id,
                                  unsigned long long capture_id) {
  std::lock_guard<std::mutex> lock(CaptureEventMutex());
  CaptureEvents().erase(CaptureEventKey(device_id, capture_id));
}

}  // namespace musa
}  // namespace tensorflow
