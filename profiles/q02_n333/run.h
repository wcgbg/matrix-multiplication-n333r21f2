#pragma once

#include "core/certificate.pb.h"
#include "profiles/q02_n333/options.h"

namespace profiles::q02_n333 {

int Run(const pb::Certificate &cert, const Options &options);

} // namespace profiles::q02_n333
