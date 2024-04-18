#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/scaled_dot_product_attention_ops.h>

namespace at {


// aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> Tensor
inline at::Tensor scaled_dot_product_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_mask={}, double dropout_p=0.0, bool is_causal=false, c10::optional<double> scale=c10::nullopt) {
    return at::_ops::scaled_dot_product_attention::call(query, key, value, attn_mask, dropout_p, is_causal, scale);
}

}
