import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from smpl_util.serialization import load_model

MAX_VALUE = 6889  # 최종 선택 가능한 최대값

SMPL_MODEL = load_model('./models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
SMPL_MODEL.pose[0] = np.pi


def draw_smpl_verices(range_min, range_max, angles):
    fig = plt.figure(figsize=(40, 16))
    kpt_color = ['grey'] * 6890
    if range_min != -1 or range_max != -1:
        kpt_color[range_min:range_max+1] = ['red'] * (range_max - range_min + 1)
    axis_limit = 1.2
    x_c = SMPL_MODEL[:, 0].mean()
    y_c = SMPL_MODEL[:, 1].mean()
    z_c = SMPL_MODEL[:, 2].mean()

    def draw_subplot(loc, elev, azim, roll):
        ax = fig.add_subplot(loc, projection='3d')
        ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis='y')
        ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
        ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
        ax.set_zlim3d([min(0, z_c - axis_limit / 2), z_c + axis_limit / 2])
        x_3d, y_3d, z_3d = SMPL_MODEL[:, 0], SMPL_MODEL[:, 1], SMPL_MODEL[:, 2]
        ax.scatter(x_3d, y_3d, z_3d, marker='o', c=kpt_color)

    draw_subplot(121, *angles[0])
    draw_subplot(122, *angles[1])

    return fig


def update_plot(thousand, hundred, tens, ones, angles):
    """
    드롭다운의 현재 선택을 바탕으로 get_range를 호출하고,
    그 결과를 plot_range_func의 입력으로 전달하여 Figure를 반환합니다.
    """
    rmin, rmax = get_range(thousand, hundred, tens, ones)
    fig = draw_smpl_verices(rmin, rmax, angles)
    return fig


def create_update_func(unit_flag):
    def update_func(thousand, hundred, tens):
        return update_dropdowns(unit_flag, thousand, hundred, tens)
    return update_func


def update_dropdowns(unit_flag, thousand, hundred, tens):
    """
    단일 업데이트 함수.
    unit_flag에 따라 이후 드롭다운(백/십/일)의 상태를 업데이트합니다.

    인자:
      - unit_flag: "thousand", "hundred", "tens" 중 하나
      - thousand, hundred, tens, ones: 현재 각 드롭다운의 값
    반환값:
      - unit_flag에 따라 업데이트할 드롭다운 컴포넌트들의 gr.update 객체들을 튜플로 반환
    """
    if unit_flag == "thousand":
        # 천의 자리(dd1) 변경 시: 백(dd2), 십(dd3), 일(dd4) 모두 재설정
        if thousand == "No All" or thousand is None:
            update_centi = gr.update(value="All", interactive=False)
            update_deci = gr.update(value="All", interactive=False)
            update_bi = gr.update(value="All", interactive=False)
        else:
            # 천의 자리 값이 정해졌으면 백의 자리 활성화 및 선택지 업데이트
            if thousand == 6000:
                # 6000이면 6000+900=6900 > 6890 이므로 900은 제외 (800은 허용)
                choices_hundred = ["All", 0, 100, 200, 300, 400, 500, 600, 700, 800]
            else:
                choices_hundred = ["All", 0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
            update_centi = gr.update(interactive=True, choices=choices_hundred, value="All")
            update_deci = gr.update(value="All", interactive=False)
            update_bi = gr.update(value="All", interactive=False)
        return update_centi, update_deci, update_bi

    elif unit_flag == "hundred":
        # 백의 자리(dd2) 변경 시: 십(dd3)와 일(dd4) 재설정
        if hundred == "All" or hundred is None:
            update_deci = gr.update(value="All", interactive=False)
            update_bi = gr.update(value="All", interactive=False)
        else:
            # 현재 천+백의 합에 따라 허용 가능한 tens 값 결정
            allowed_max_tens = MAX_VALUE - (thousand + hundred)
            # 기본 tens 옵션: 0,10,...,90
            default_tens = list(range(0, 100, 10))
            # allowed_max_tens보다 작거나 같은 값만 선택지에 포함
            tens_options = [x for x in default_tens if x <= allowed_max_tens]
            tens_choices = ["All"] + tens_options
            update_deci = gr.update(interactive=True, choices=tens_choices, value="All")
            update_bi = gr.update(value="All", interactive=False)
        return update_deci, update_bi

    elif unit_flag == "tens":
        # 십의 자리(dd3) 변경 시: 일(dd4)만 업데이트
        if tens == "All" or tens is None:
            update_bi = gr.update(value="All", interactive=False)
        else:
            current_sum = thousand + hundred + tens
            allowed_max = MAX_VALUE - current_sum
            max_digit = min(9, allowed_max)  # 일의 자리는 0~9 범위 내에서 결정
            ones_choices = ["All"] + list(range(0, max_digit + 1))
            update_bi = gr.update(interactive=True, choices=ones_choices, value="All")
        return update_bi


# 드롭다운의 현재 선택 상태에 따른 최소값, 최대값을 반환하는 함수
def get_range(thousand, hundred, tens, ones):
    if thousand == "No All" or thousand is None:
        return -1, -1
    t = thousand
    if hundred == "All" or hundred is None:
        if t == 6000:
            lower, upper = t, MAX_VALUE
        else:
            lower, upper = t, t + 999
    elif tens == "All" or tens is None:
        lower, upper = t + hundred, t + hundred + 99
    elif ones == "All" or ones is None:
        lower, upper = t + hundred + tens, t + hundred + tens + 9
    else:
        lower = upper = t + hundred + tens + ones
    if upper > MAX_VALUE:
        upper = MAX_VALUE
    return lower, upper


with gr.Blocks(title="3D Mesh Vertices visualization") as demo:
    with gr.Row():
        milli = gr.Dropdown(
            choices=['No All', 0, 1000, 2000, 3000, 4000, 5000, 6000],
            label='Thousands',
            interactive=True
        )
        centi = gr.Dropdown(
            choices=['All'],
            label='Hundreds',
            interactive=False
        )
        deci = gr.Dropdown(
            choices=["All"],
            label="Tens",
            value="All",
            interactive=False
        )
        bi = gr.Dropdown(
            choices=["All"],
            label="Ones",
            value="All",
            interactive=False
        )

    # 현재 선택 범위(최소, 최대)를 표시할 컴포넌트
    range_min = gr.Number(label="시작 인덱스", value=-1)
    range_max = gr.Number(label="마지막 인덱스", value=-1)

    _angles1 = gr.State([[1, 0, 0], [1, 180, 0]])
    _fig = draw_smpl_verices(range_min.value, range_max.value, _angles1.value)
    plot_output_1 = gr.Plot(_fig, label="Front / Back")
    _angles2 = gr.State([[45, 0, 0], [45, 180, 0]])
    _fig = draw_smpl_verices(range_min.value, range_max.value, _angles2.value)
    plot_output_2 = gr.Plot(_fig, label="Overview Front / Back")
    _angles3 = gr.State([[1, 45, 0], [1, -45, 0]])
    _fig = draw_smpl_verices(range_min.value, range_max.value, _angles3.value)
    plot_output_3 = gr.Plot(_fig, label="Front Side 45'/-45'")
    _angles4 = gr.State([[1, 135, 0], [1, -135, 0]])
    _fig = draw_smpl_verices(range_min.value, range_max.value, _angles4.value)
    plot_output_4 = gr.Plot(_fig, label="Back Side 135'/-135'")

    milli.change(
        fn=create_update_func("thousand"),
        inputs=[milli, centi, deci],
        outputs=[centi, deci, bi]
    )
    centi.change(
        fn=create_update_func("hundred"),
        inputs=[milli, centi, deci],
        outputs=[deci, bi]
    )
    deci.change(
        fn=create_update_func("tens"),
        inputs=[milli, centi, deci],
        outputs=bi
    )

    # 드롭다운이 변경될 때마다 현재 선택 범위를 업데이트하여 최소, 최대 값을 표시
    for comp in [milli, centi, deci, bi]:
        comp.change(fn=get_range, inputs=[milli, centi, deci, bi],
                    outputs=[range_min, range_max])
        comp.change(fn=update_plot, inputs=[milli, centi, deci, bi, _angles1], outputs=plot_output_1)
        comp.change(fn=update_plot, inputs=[milli, centi, deci, bi, _angles2], outputs=plot_output_2)
        comp.change(fn=update_plot, inputs=[milli, centi, deci, bi, _angles3], outputs=plot_output_3)
        comp.change(fn=update_plot, inputs=[milli, centi, deci, bi, _angles4], outputs=plot_output_4)


if __name__ == "__main__":
    demo.launch(share=False)