#ifndef TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_
#define TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_

#include <musa_runtime.h>

#include <cstddef>
#include <vector>

namespace tensorflow {
namespace musa {

// Capture-scoped event ownership. Events are retained while a stream capture
// is active and claimed by the graph owner before the graph is released.
void RetainCaptureDependencyEvent(musaEvent_t event,
                                  unsigned long long capture_id);
std::vector<musaEvent_t> TakeCaptureDependencyEvents(
    unsigned long long capture_id);
std::size_t PendingCaptureDependencyEventCount(
    unsigned long long capture_id);

}  // namespace musa
}  // namespace tensorflow

#endif  // TENSORFLOW_MUSA_MU_DEVICE_MUSA_CAPTURE_EVENT_REGISTRY_H_
