#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"
// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim));
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, int64_t numel) {
  if (tensor.numel() != numel) {
    std::stringstream err;
    err << "incorrect numel for input tensor. expected " << numel << ", got " << tensor.numel();
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/BinaryOps.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/types.h>
#include <ATen/ops/bernoulli_native.h>

#define reinterpret_tensor torch::inductor::_reinterpret_tensor
#define alloc_from_pool torch::inductor::_alloc_from_pool
#include <c10/util/generic_math.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#include "/var/folders/zs/vx_6lqwd1cx250x_47lxr10r0000gn/T/torchinductor_scottpyle/wy/cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused_0(float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = static_cast<long>(2);
            auto tmp2 = tmp0 < tmp1;
            auto tmp3 = static_cast<long>(1);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = static_cast<float>(-0.05833464115858078);
            auto tmp6 = static_cast<float>(-0.1370282918214798);
            auto tmp7 = tmp4 ? tmp5 : tmp6;
            auto tmp8 = static_cast<long>(3);
            auto tmp9 = tmp0 < tmp8;
            auto tmp10 = static_cast<float>(0.1504911333322525);
            auto tmp11 = static_cast<float>(-0.23632794618606567);
            auto tmp12 = tmp9 ? tmp10 : tmp11;
            auto tmp13 = tmp2 ? tmp7 : tmp12;
            out_ptr0[static_cast<long>(x0)] = tmp13;
        }
    }
}

#include "/var/folders/zs/vx_6lqwd1cx250x_47lxr10r0000gn/T/torchinductor_scottpyle/wy/cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused_convolution_max_pool2d_with_indices_relu_1(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>((2L*x2) + (256L*x1) + (16384L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(1L + (2L*x2) + (256L*x1) + (16384L*x0))];
                    auto tmp3 = in_ptr0[static_cast<long>(128L + (2L*x2) + (256L*x1) + (16384L*x0))];
                    auto tmp5 = in_ptr0[static_cast<long>(129L + (2L*x2) + (256L*x1) + (16384L*x0))];
                    auto tmp2 = max_propagate_nan(tmp1, tmp0);
                    auto tmp4 = max_propagate_nan(tmp3, tmp2);
                    auto tmp6 = max_propagate_nan(tmp5, tmp4);
                    auto tmp7 = std::max(tmp6, decltype(tmp6)(0));
                    out_ptr0[static_cast<long>(x0 + (4L*x2) + (256L*x1))] = tmp7;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (36L*x0))];
                    out_ptr1[static_cast<long>(x1 + (4L*x2) + (36L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            auto tmp1 = static_cast<long>(4);
            auto tmp2 = tmp0 < tmp1;
            auto tmp3 = static_cast<long>(2);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = static_cast<long>(1);
            auto tmp6 = tmp0 < tmp5;
            auto tmp7 = static_cast<float>(0.1629398614168167);
            auto tmp8 = static_cast<float>(-0.11850599199533463);
            auto tmp9 = tmp6 ? tmp7 : tmp8;
            auto tmp10 = static_cast<long>(3);
            auto tmp11 = tmp0 < tmp10;
            auto tmp12 = static_cast<float>(-0.044845741242170334);
            auto tmp13 = static_cast<float>(0.05762872472405434);
            auto tmp14 = tmp11 ? tmp12 : tmp13;
            auto tmp15 = tmp4 ? tmp9 : tmp14;
            auto tmp16 = static_cast<long>(6);
            auto tmp17 = tmp0 < tmp16;
            auto tmp18 = static_cast<long>(5);
            auto tmp19 = tmp0 < tmp18;
            auto tmp20 = static_cast<float>(0.09651754796504974);
            auto tmp21 = static_cast<float>(0.14812764525413513);
            auto tmp22 = tmp19 ? tmp20 : tmp21;
            auto tmp23 = static_cast<long>(7);
            auto tmp24 = tmp0 < tmp23;
            auto tmp25 = static_cast<float>(-0.023242950439453125);
            auto tmp26 = static_cast<float>(-0.06401248276233673);
            auto tmp27 = tmp24 ? tmp25 : tmp26;
            auto tmp28 = tmp17 ? tmp22 : tmp27;
            auto tmp29 = tmp2 ? tmp15 : tmp28;
            out_ptr2[static_cast<long>(x0)] = tmp29;
        }
    }
}

#include "/var/folders/zs/vx_6lqwd1cx250x_47lxr10r0000gn/T/torchinductor_scottpyle/wy/cwyvgno7oj63mpe36f4v6pizgeyvccmavffogp6xnqv56a32gbwo.h"
extern "C" void cpp_fused_addmm_2(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    {
        {
            float tmp_acc0 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (c10::div_floor_integer(x0, 256L)))];
                auto tmp1 = in_ptr0[static_cast<long>(8L + (16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (c10::div_floor_integer(x0, 256L)))];
                auto tmp3 = in_ptr0[static_cast<long>(256L + (16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (c10::div_floor_integer(x0, 256L)))];
                auto tmp5 = in_ptr0[static_cast<long>(264L + (16L*(static_cast<long>(x0) % static_cast<long>(16L))) + (512L*(static_cast<long>(c10::div_floor_integer(x0, 16L)) % static_cast<long>(16L))) + (c10::div_floor_integer(x0, 256L)))];
                auto tmp8 = in_ptr1[static_cast<long>(x0)];
                auto tmp2 = max_propagate_nan(tmp1, tmp0);
                auto tmp4 = max_propagate_nan(tmp3, tmp2);
                auto tmp6 = max_propagate_nan(tmp5, tmp4);
                auto tmp7 = std::max(tmp6, decltype(tmp6)(0));
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                tmp_acc0 = tmp_acc0 + tmp9;
            }
            out_ptr0[static_cast<long>(0L)] = tmp_acc0;
        }
    }
    {
        auto tmp0 = out_ptr0[static_cast<long>(0L)];
        auto tmp1 = static_cast<float>(1.0);
        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
        auto tmp3 = static_cast<float>(0.014650809578597546);
        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
        in_out_ptr0[static_cast<long>(0L)] = tmp4;
    }
}
namespace torch {
namespace aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
};
}  // namespace

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1, 1, 3, device_str, cubin_dir) {
    inputs_info_[0].name = "arg6_1";
    constants_info_[0].name = "L__self___conv1_weight";
    constants_info_[0].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 144;
    constants_info_[0].from_folded = false;
    constants_info_[0].shape = {4, 1, 3, 3};
    constants_info_[0].stride = {9, 9, 3, 1};
    constants_info_[0].original_fqn = "conv1.weight";
    constants_info_[1].name = "L__self___conv2_weight";
    constants_info_[1].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 1152;
    constants_info_[1].from_folded = false;
    constants_info_[1].shape = {8, 4, 3, 3};
    constants_info_[1].stride = {36, 9, 3, 1};
    constants_info_[1].original_fqn = "conv2.weight";
    constants_info_[2].name = "L__self___fc_weight";
    constants_info_[2].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 8192;
    constants_info_[2].from_folded = false;
    constants_info_[2].shape = {1, 2048};
    constants_info_[2].stride = {2048, 1};
    constants_info_[2].original_fqn = "fc.weight";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": null, \"context\": null, \"children_spec\": []}]";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {

    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, 1);
    auto arg6_1 = std::move(inputs[0]);
    auto L__self___conv1_weight = *tensor_handle_to_tensor_pointer(constants_->at(0));
    auto L__self___conv2_weight = *tensor_handle_to_tensor_pointer(constants_->at(1));
    auto L__self___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(2));
    inputs.clear();
    auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    at::Tensor buf0 = at::detail::empty_strided_cpu({4L, }, {1L, }, at::kFloat);
    cpp_fused_0((float*)(buf0.data_ptr()));
    // Source Nodes: [l__self___conv1], Original ATen: [aten.convolution]
    auto buf1 = at::convolution(arg6_1, L__self___conv1_weight, buf0, {2LL, 2LL}, {1LL, 1LL}, {1LL, 1LL}, false, {0LL, 0LL}, 1LL);
    arg6_1.reset();
    buf0.reset();
    at::Tensor buf2 = at::detail::empty_strided_cpu({1L, 4L, 64L, 64L}, {16384L, 1L, 256L, 4L}, at::kFloat);
    at::Tensor buf3 = at::detail::empty_strided_cpu({8L, 4L, 3L, 3L}, {36L, 1L, 12L, 4L}, at::kFloat);
    at::Tensor buf4 = at::detail::empty_strided_cpu({8L, }, {1L, }, at::kFloat);
    cpp_fused_convolution_max_pool2d_with_indices_relu_1((float*)(buf1.data_ptr()), (float*)(L__self___conv2_weight.data_ptr()), (float*)(buf2.data_ptr()), (float*)(buf3.data_ptr()), (float*)(buf4.data_ptr()));
    buf1.reset();
    // Source Nodes: [l__self___conv2, l__self___pool1, x], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
    auto buf5 = at::convolution(buf2, buf3, buf4, {2LL, 2LL}, {1LL, 1LL}, {1LL, 1LL}, false, {0LL, 0LL}, 1LL);
    buf2.reset();
    buf3.reset();
    buf4.reset();
    at::Tensor buf6 = at::detail::empty_strided_cpu({1L, }, {1L, }, at::kFloat);
    decltype(auto) buf7 = reinterpret_tensor(buf6, {1L, 1L}, {1L, 1L}, 0L); buf6.reset();  // reuse
    cpp_fused_addmm_2((float*)(buf7.data_ptr()), (float*)(buf5.data_ptr()), (float*)(L__self___fc_weight.data_ptr()));
    output_handles[0] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf7));
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch
