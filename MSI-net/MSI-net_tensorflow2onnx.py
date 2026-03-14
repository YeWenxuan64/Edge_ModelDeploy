import os
import numpy as np
import onnx
import onnxslim

import tensorflow as tf
from tf2onnx.tf_loader import from_saved_model
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx import optimizer


current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)


tensorflow_savemodel_path = 'models_convert/original/save_model'
tensorflow_savemodel_path = os.path.join(current_path, tensorflow_savemodel_path)

intermediate_onnx_model_path = 'models_convert/onnx/MSI-net_from_tf.onnx'
intermediate_onnx_model_path = os.path.join(current_path, intermediate_onnx_model_path)

final_onnx_model_path = 'models_convert/onnx/MSI-net_[1,3,160,320].onnx'
final_onnx_model_path = os.path.join(current_path, final_onnx_model_path)


target_conv_list = [('PartitionedCall/model/layer_from_saved_model/PartitionedCall/aspp/conv1_2/BiasAdd', 10),
                    ('PartitionedCall/model/layer_from_saved_model/PartitionedCall/aspp/conv1_3/BiasAdd', 40),
                    ('PartitionedCall/model/layer_from_saved_model/PartitionedCall/aspp/conv1_4/BiasAdd', 80),]

target_reducemax_node_name = 'PartitionedCall/model/layer_from_saved_model/PartitionedCall/Max'


def convert_common(frozen_graph, name="unknown", output_path=None, **kwargs) -> onnx.ModelProto:
    """Common processing for conversion."""
    model_proto = None
    external_tensor_storage = None
    const_node_values = None
    large_model = False

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_graph, name='')
        g = process_tf_graph(tf_graph, const_node_values=const_node_values, **kwargs)
        
        onnx_graph = optimizer.optimize_graph(g, not large_model)
        model_proto = onnx_graph.make_model(name, external_tensor_storage=external_tensor_storage)
    
    return model_proto

def modify_normalization(onnx_model:onnx.ModelProto):
    graph = onnx_model.graph

    # 修改输出归一化
    target_reducemax_node = None
    target_reducemax_output_shape = None
    for i, node in enumerate(graph.node):
        if node.name == target_reducemax_node_name:
            target_reducemax_node = node
            print('found target_reducemax_node: ', target_reducemax_node.name)

            # 获取target_reducemax_node的输出形状
            for output in graph.output:
                if output.name == target_reducemax_node.output[0]:
                    target_reducemax_output_shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
                    break
            
            # 如果在graph.output中找不到，尝试在value_info中查找
            if target_reducemax_output_shape is None:
                for value_info in graph.value_info:
                    if value_info.name == target_reducemax_node.output[0]:
                        target_reducemax_output_shape = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
                        break

            if target_reducemax_output_shape is None:
                target_reducemax_output_shape = [1, 1, 1, 1]  # 默认形状

            break

    if target_reducemax_node is not None:
        # 创建常量节点255
        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['255_const'],
            value=onnx.helper.make_tensor(name='255_const_tensor', data_type=onnx.TensorProto.FLOAT, dims=target_reducemax_output_shape, vals=[255.0]),
            name='255_const_node')

        # 创建除法节点
        div_node = onnx.helper.make_node(
            'Div',
            inputs=[target_reducemax_node.output[0], const_node.output[0]],
            outputs=[target_reducemax_node.output[0] + '_div'],
            name=target_reducemax_node.name + '_div')


        for i, node in enumerate(graph.node):
            if node.input:
                break_out = False

                for j, input_name in enumerate(node.input):
                    if input_name == target_reducemax_node.output[0]:
                        node.input[j] = div_node.output[0]
                        break_out = True
                        print('modified node: ', node.name)
                        break

                if break_out:
                    break

        # 将新节点插入到图中
        graph.node.insert(i + 1, div_node)
        graph.node.insert(i + 1, const_node)

    return onnx_model

def export():
    model_path = tensorflow_savemodel_path
    onnx_output_path = intermediate_onnx_model_path
    onnx_opset = 11
    model_name = os.path.basename(intermediate_onnx_model_path)
    new_tf_input_shape = [1, 160, 320, 3]



    graph_def, inputs, outputs, initialized_tables, tensors_to_rename = from_saved_model(
        model_path, 
        input_names=None, 
        output_names=None, 
        return_initialized_tables=True, 
        return_tensors_to_rename=True, 
        use_graph_names=False)
    
    
    tensors_to_rename:dict
    dict_iter = iter(tensors_to_rename)
    print('inputs and outputs tensors name: ', tensors_to_rename)

    input_name = next(dict_iter)
    output_name = next(dict_iter)

    input_names = [str(input_name)]
    output_names = [str(output_name)]

    tensors_to_rename[input_name] = "input"
    tensors_to_rename[output_name] = "output"

    shape_override = {input_name: new_tf_input_shape}
    print('overrided tensorflow model input shape: ', shape_override)

    with tf.device("/cpu:0"):
        model_proto = convert_common(
            graph_def,
            name=model_name,
            opset=onnx_opset,
            shape_override=shape_override,
            input_names=inputs,
            output_names=outputs,
            inputs_as_nchw = input_names,
            outputs_as_nchw = output_names,
            tensors_to_rename=tensors_to_rename,
            output_path=onnx_output_path)

    onnx_model = modify_normalization(model_proto)

    onnx_model = onnxslim.slim(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, onnx_output_path)


def modify_conv():
    model = onnx.load_model(intermediate_onnx_model_path)

    # 获取图
    graph = model.graph

    # 遍历目标卷积列表
    for index, (target_conv_name, num_splits) in enumerate(target_conv_list):
        print('turns:', index + 1, 'of', len(target_conv_list))

        # 找到目标卷积节点
        target_conv = None
        target_conv_index = 0
        for i, node in enumerate(graph.node):
            if node.name == target_conv_name:
                target_conv = node
                target_conv_index = i
                break
        print('find target conv: ', target_conv.name)


        # 找到输入和输出节点
        input_concat = None
        output_relu = None

        # 找到输入Concat节点
        for node in graph.node:
            if node.output[0] == target_conv.input[0]:
                input_concat = node
                break
        print("find target conv's input concat: ", input_concat.name)

        # 找到输出ReLU节点
        for i, node in enumerate(graph.node):
            if node.input[0] == target_conv.output[0]:
                output_relu = node
                break
        print("find target conv's output relu: ", output_relu.name)


        # 获取卷积权重和偏置
        target_conv_weight = None
        target_conv_weight_index = 0
        conv_bias = None
        for i, initializer in enumerate(graph.initializer):
            if initializer.name == target_conv.input[1]:
                target_conv_weight = initializer
                target_conv_weight_index = i

            elif initializer.name == target_conv.input[2]:
                conv_bias = initializer
        print("find target conv's weight: ", target_conv_weight.name, target_conv_weight.dims)
        print("find target conv's bias: ", conv_bias.name)


        # 拆分权重
        weight_shape = target_conv_weight.dims
        split_size = weight_shape[1] // num_splits
        split_weights:list[onnx.TensorProto] = []
        print('split num: ', num_splits, 'split size: ', split_size)

        for i in range(num_splits):
            start = i * split_size
            end = start + split_size if i < num_splits - 1 else weight_shape[1]
            
            # 创建新的权重
            new_weight = onnx.numpy_helper.to_array(target_conv_weight)
            new_weight = new_weight[:, start:end, :, :]
            new_weight_initializer = onnx.numpy_helper.from_array(new_weight)
            new_weight_initializer.name = f"{target_conv.name}_weight_tile_{i}"

            split_weights.append(new_weight_initializer)
            print('splited conv weight: ', new_weight_initializer.name, new_weight_initializer.dims)




        # 创建拆分后的卷积节点
        for attribute in target_conv.attribute:
            if attribute.name == 'dilations':
                dilations = attribute.ints
                
            elif attribute.name == 'kernel_shape':
                kernel_shape = attribute.ints

            elif attribute.name == 'pads':
                pads = attribute.ints

            elif attribute.name == 'strides':
                strides = attribute.ints

        print('kernel_shape: ', kernel_shape, 'dilations: ', dilations, 'pads: ', pads, 'strides: ', strides)

        conv_nodes:list[onnx.NodeProto] = []
        for i, weight in enumerate(split_weights):
            new_conv_node_inputs = [f"{target_conv.name}_split_{i}", weight.name]
            if i == 0:
                new_conv_node_inputs.append(conv_bias.name)

            conv_node = onnx.helper.make_node(
                'Conv',
                inputs=new_conv_node_inputs,
                outputs=[f"{target_conv.name}_out_{i}"],
                name=f"{target_conv.name}_tile_{i}",
                dilations = dilations,
                kernel_shape = kernel_shape,
                pads = pads,
                strides = strides
            )

            conv_nodes.append(conv_node)
            print('splited conv node: ', conv_node.name)


        # 创建新的Split初始化器
        split_initializer = onnx.helper.make_tensor(
            name = f"{target_conv.name}_split_init", 
            data_type = onnx.TensorProto.INT64, 
            dims = [num_splits], 
            vals = np.array([split_size] * num_splits, dtype=np.int64).tobytes(),
            raw = True  # 添加raw参数
        )


        # 创建新的Split节点
        split_node = onnx.helper.make_node(
            'Split',
            inputs=[target_conv.input[0], split_initializer.name],
            outputs=[new_conv.input[0] for new_conv in conv_nodes],
            name=f"{target_conv.name}_split",
            axis=1
        )
        print('added new split node: ', split_node.name)


        # 创建Add节点
        add_nodes:list[onnx.NodeProto] = []
        add_inputs = [conv_nodes[0].output[0]]
        for i in range(1, num_splits):
            add_node = onnx.helper.make_node(
                'Add',
                inputs=add_inputs + [conv_nodes[i].output[0]],
                outputs=[f"{target_conv.name}_add_out_{i}"],
                name=f"{target_conv.name}_add_{i}"
            )

            add_nodes.append(add_node)
            add_inputs = [add_node.output[0]]
            print('added new add node: ', add_node.name)


        # 插入新的Split节点
        graph.node.insert(target_conv_index, split_node)

        # 插入新的卷积节点
        for i, conv_node in enumerate(conv_nodes):
            graph.node.insert(target_conv_index + i + 1, conv_node)

        # 插入新的Add节点
        for i, add_node in enumerate(add_nodes):
            graph.node.insert(target_conv_index + i + 1 + len(conv_nodes), add_node)

        # 更新ReLU节点的输入
        output_relu.input[0] = add_nodes[-1].output[0]

        # 插入新的Split初始化器
        graph.initializer.insert(target_conv_index - 1, split_initializer)

        # 插入新的卷积权重
        for i, weight in enumerate(split_weights):
            graph.initializer.insert(target_conv_weight_index + i + 1, weight)

        # 删除原始的卷积节点
        graph.node.remove(target_conv)

        # 删除原始的卷积权重
        graph.initializer.remove(target_conv_weight)




    new_model = onnx.helper.make_model(model.graph, producer_name=model.producer_name, opset_imports=[onnx.helper.make_opsetid("", 15)])
    
    new_model = onnx.shape_inference.infer_shapes(new_model, check_type=True, strict_mode=True)
    onnx.save_model(new_model, final_onnx_model_path)


if __name__ == '__main__':
    export()
    modify_conv()

