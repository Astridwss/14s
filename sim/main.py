import json
from plan_file_process import PlanFileProcess
from trainingenvironment import TrainingEnv
from datastruct import AgentObservation, AgentActionCommand


if __name__ == '__main__':
    file_path = './PlanFile/预警任务-印度_20260313155241.json'
    # plan_file_process = PlanFileProcess()
    # battle_scene = plan_file_process.read_battle_scene_from_json(plan_id=867, file_path=file_path)
    # plan_result = plan_file_process.read_plan_result_from_json(plan_id=867, file_path=file_path)
    # plan_file_info = plan_file_process.read_plan_file_info_from_json(plan_id=867, file_path=file_path)
    # plan_file_process.write_plan_file_info_to_csv(plan_file_info, './PlanFile/il_test.csv')

    training_env = TrainingEnv()
    training_env.load_battle_scene(plan_id=867, file_path=file_path)
    agent_action = AgentActionCommand()
    observation, flag = training_env.step_forward(agent_action=agent_action)
    observation, flag = training_env.step_forward(agent_action=agent_action)
    training_env.reset()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            print("JSON 文件解析成功，格式为dict")

            str = data['planInfoList'][0]['splitQuduanResult']
            inner_data = json.loads(str)

            print("missileCalc解析成功")

        elif isinstance(data, list):
            print("JSON 文件解析成功，格式为list")
        else:
            print("JSON 文件格式未知")

    except FileNotFoundError:
        print("文件未找到")
    except json.JSONDecodeError as e:
        print("JSON 解析失败")
    except Exception as e:
        print(f"发生未知错误：{e}")

