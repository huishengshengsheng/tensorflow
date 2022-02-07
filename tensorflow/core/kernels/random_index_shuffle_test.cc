/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/random_index_shuffle.h"

#include <array>
#include <vector>

#include "testing/base/public/benchmark.h"
#include <gmock/gmock.h>
#include "testing/base/public/gunit.h"
#include "absl/base/integral_types.h"

namespace tensorflow {
namespace random {
namespace {

class RandomIndexShuffleTest : public testing::TestWithParam<uint64_t> {
 public:
  uint64_t GetMaxValue() const { return GetParam(); }
};

// Check that we do a correct bijection.
TEST_P(RandomIndexShuffleTest, Bijection) {
  const std::array<uint32, 3>& key = {42, 73, 1991};
  const uint64_t max_value = GetMaxValue();
  std::vector<bool> seen(max_value + 1, false);
  for (uint64_t value = 0; value <= max_value; ++value) {
    const uint64 output_value = index_shuffle(value, key, max_value);
    EXPECT_GE(output_value, 0);
    EXPECT_LE(output_value, max_value);
    EXPECT_FALSE(seen[output_value]);
    seen[output_value] = true;
  }
}

INSTANTIATE_TEST_SUITE_P(MaxValueTests, RandomIndexShuffleTest,
                         testing::Values(285, 17, 23495, 499'000));

void BM_ThroughputSmall(benchmark::State& state) {
  const std::array<uint32, 3>& key = {42, 73, 1991};
  const uint64_t max_value = 42'982;
  uint64_t value = 0;
  for (auto s : state) {
    const auto output_value = index_shuffle(value, key, max_value);
    EXPECT_GE(output_value, 0);
    EXPECT_LE(output_value, max_value);
    value = (value + 1) % (max_value + 1);
  }
}

void BM_ThroughputBig(benchmark::State& state) {
  const std::array<uint32, 3>& key = {42, 73, 1991};
  const uint64_t max_value = 814'982'943;
  uint64_t value = 0;
  for (auto s : state) {
    const auto output_value = index_shuffle(value, key, max_value);
    EXPECT_GE(output_value, 0);
    EXPECT_LE(output_value, max_value);
    value = (value + 1) % (max_value + 1);
  }
}

BENCHMARK(BM_ThroughputSmall);
BENCHMARK(BM_ThroughputBig);

}  // namespace
}  // namespace random
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  RunSpecifiedBenchmarks();
  return RUN_ALL_TESTS();
}
