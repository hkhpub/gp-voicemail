import numpy as np
from task_presentation import VoiceTask
# from task_presentation import VoiceTask


if __name__ == '__main__':

    env_file = 'examples/env/voicemail.pomdp'

    task = VoiceTask(env_file, np.array([[0.65], [0.35]]))

    task.do_steps(1000)
    task.print_summary()

    # task.test_get_best_action()
    # task.test_init_episode()
    # task.test_get_observation_str()

