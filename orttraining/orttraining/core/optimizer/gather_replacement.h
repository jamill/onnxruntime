// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// gather_replacement.h

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GatherReplacement

Rewrite rule that replaces Gather with GatherInternal, that has additional outputs
that are precomputed for use in executing GatherGrad more efficiently.

*/
class GatherReplacement : public RewriteRule {
 public:
  GatherReplacement() noexcept : RewriteRule("GatherReplacement") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Gather"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
