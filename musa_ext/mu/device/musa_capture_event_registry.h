#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_

#include <musa_runtime.h>

#include <cstddef>
#include <vector>

namespace tensorflow {
namespace musa {

// Capture-scoped event ownership. Events are retained while a stream capture
// is active and claimed by the graph owner before the graph is released.
bool RetainCaptureDependencyEvent(musaEvent_t event, int device_id,
                                  unsigned long long capture_id);
std::vector<musaEvent_t> TakeCaptureDependencyEvents(
    int device_id, unsigned long long capture_id);
std::size_t PendingCaptureDependencyEventCount(
    int device_id, unsigned long long capture_id);
void ReleaseCaptureDependencyGate(int device_id,
                                  unsigned long long capture_id);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_
