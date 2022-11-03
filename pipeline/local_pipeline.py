from typing import Dict, Optional, Text

from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.v1.components import ImportSchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Pusher
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    schema_path: Text,
    modules: Dict[Text, Text],
    serving_model_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train-*.tfrec"),
            example_gen_pb2.Input.Split(name="eval", pattern="test-*.tfrec"),
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
    components.append(example_gen)

    statistics_gen = StatisticsGen(examples= example_gen.outputs['examples'])
    components.append(statistics_gen)

    schema_gen = ImportSchemaGen(schema_file = schema_path)
    components.append(schema_gen)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    components.append(example_validator) 

    transform_args = {
        "examples": example_gen.outputs['examples'],
        "schema" : schema_gen.outputs['schema'],
        "preprocessing_fn" : modules['preprocessing_fn'],
    }
    transform = Transform(**transform_args)
    components.append(transform)

    trainer_args = {
        "run_fn": modules["training_fn"],
        # "transformed_examples": transform.outputs["transformed_examples"],
        "examples":transform.outputs["transformed_examples"],
        "transform_graph": transform.outputs["transform_graph"],
        "schema": schema_gen.outputs["schema"],
    }
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    pusher_args = {
        "model": trainer.outputs["model"],
        "push_destination": tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    }
    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )
