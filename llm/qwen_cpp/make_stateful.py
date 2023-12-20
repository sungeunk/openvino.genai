from openvino._offline_transformations import apply_make_stateful_transformation
from openvino.runtime import Core
import openvino as ov
from openvino import save_model

model_name = "NNCF_INT4_SYM_PURE\\modified_openvino_model.xml"
stateful_model_name = "NNCF_INT4_SYM_PURE\\modified_stateful_openvino_model.xml"
core = Core()
ov_model = core.read_model(model_name)
past_kv_count = 32
input_output_map = {}
past_kv_list = []
for idx in range(past_kv_count):
    input_output_map[f"past_key_values.{idx}.key"] = f"present.{idx}.key"
    input_output_map[f"past_key_values.{idx}.value"] = f"present.{idx}.value"
    past_kv_list.append(f"past_key_values.{idx}.key")
    past_kv_list.append(f"past_key_values.{idx}.value")

print("input_output_map: ", input_output_map)
print("past_kv_list: ", past_kv_list)

print(ov_model.input)
for input_name in past_kv_list:
    input = ov_model.input(input_name)
    shape = input.get_partial_shape()
    shape[0] = 1
    input.get_node().set_partial_shape(shape)
 
ov_model.validate_nodes_and_infer_types()
apply_make_stateful_transformation(ov_model, input_output_map)
print("OpenVINO model stateful input:: ", ov_model.input)
print("OpenVINO model stateful output: ",  ov_model.output)
print("Save stateful model to disk")
save_model(ov_model, stateful_model_name, compress_to_fp16=True)
