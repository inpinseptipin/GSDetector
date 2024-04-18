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



#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_ops.h>

namespace at {


// aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
inline ::std::tuple<at::Tensor,at::Tensor> _scaled_dot_product_flash_attention_for_cpu(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p=0.0, bool is_causal=false, const ::std::optional<at::Tensor> & attn_mask={}, ::std::optional<double> scale=::std::nullopt) {
    return at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(query, key, value, dropout_p, is_causal, attn_mask, scale);
}

}
