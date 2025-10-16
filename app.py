import json
from itertools import combinations
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import Select, Slider, Button, ColumnDataSource, CustomJS, Div, CheckboxGroup
from bokeh.embed import file_html
from bokeh.resources import CDN


def create_groups_config():
    """
    Generates the configuration for all simulation groups.
    """
    group_prefixes = ['A', 'B', 'C']
    group_numbers = range(1, 6)
    all_groups = []
    for p in group_prefixes:
        for n in group_numbers:
            all_groups.append(f"{p}{n}")
            all_groups.append(f"{p}{n}*")

    E6_VALUES_uF = [10, 15, 22, 33, 47, 68]
    unique_combinations = [list(c) for c in combinations(E6_VALUES_uF, 3)][:5]

    tolerances = {'A': 0.05, 'B': 0.10, 'C': 0.20}
    noise_levels = {'normal': 0.01, 'high': 0.02}

    groups_config = {}
    for group_name in all_groups:
        prefix = group_name[0]
        number = int(group_name[1])
        is_high_noise = group_name.endswith('*')
        assigned_caps = unique_combinations[number - 1]

        groups_config[group_name] = {
            'nominal_caps_uF': sorted(assigned_caps),
            'tolerance': tolerances[prefix],
            'noise_level': noise_levels['high'] if is_high_noise else noise_levels['normal']
        }

    return groups_config, all_groups


def create_widgets(all_groups):
    """
    Creates the Bokeh widgets that make up the simulator's UI.
    """
    widgets = {}
    widgets['source_clipped'] = ColumnDataSource(data={'t': [], 'v1': [], 'v2': [], 'v3': []})
    widgets['source_raw'] = ColumnDataSource(data={'t': [], 'v1': [], 'v2': [], 'v3': []})
    widgets['full_data_source_clipped'] = ColumnDataSource(data={'t': [], 'v1': [], 'v2': [], 'v3': []})
    widgets['full_data_source_raw'] = ColumnDataSource(data={'t': [], 'v1': [], 'v2': [], 'v3': []})

    widgets['prefix_select'] = Select(title="アルファベット", value="A", options=['A', 'B', 'C'], width=100)
    widgets['number_select'] = Select(title="数字", value="1", options=[str(i) for i in range(1, 6)], width=100)

    widgets['asterisk_label'] = Div(text="<div style='font-size: 14px; font-weight: 600;'>オプション</div>")
    widgets['asterisk_checkbox'] = CheckboxGroup(labels=["*あり"], active=[], width=100)

    V0_FIXED = 5.0
    widgets['v0_display'] = Div(
        text=f"<div style='font-size: 14px; font-weight: 600;'>電源電圧 E: {V0_FIXED:.1f} V (固定)</div>")

    initial_r_value = 100_000
    initial_r_text = f"<div style='font-size: 14px; font-weight: 600;'>抵抗値 R: {initial_r_value:,} ({int(initial_r_value / 1000)} kΩ)</div>"
    widgets['r_label'] = Div(text=initial_r_text)
    widgets['R_slider'] = Slider(start=10_000, end=1_000_000, value=initial_r_value, step=10_000, title=None,
                                 show_value=False)

    widgets['start_button'] = Button(label="計測開始", button_type="success")
    widgets['download_button'] = Button(label="計測データをダウンロード (.csv)", button_type="primary", disabled=True)

    p = figure(height=400, width=600, title="CR回路 充電電圧のシミュレーション",
               x_axis_label="時刻 t [s]", y_axis_label="コンデンサ端子電圧 Vc [V]")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    subscripts = ['₁', '₂', '₃']

    for i in range(3):
        p.scatter(x='t', y=f'v{i + 1}', source=widgets['source_raw'], marker='x',
                  color=colors[i], size=5, legend_label=f"V{subscripts[i]}(t) (補正前)")

    for i in range(3):
        p.scatter(x='t', y=f'v{i + 1}', source=widgets['source_clipped'], marker='circle',
                  color=colors[i], size=4, legend_label=f"V{subscripts[i]}(t) (補正後)")

    p.legend.ncols = 2
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.title = "凡例"
    widgets['plot'] = p

    return widgets


def create_callbacks(widgets, groups_config):
    """
    Creates and attaches JavaScript callbacks to the widgets.
    """
    r_slider_callback = CustomJS(args=dict(slider=widgets['R_slider'], label=widgets['r_label']), code="""
        const value_kOhm = (slider.value / 1000).toFixed(0);
        const value_Ohm_formatted = slider.value.toLocaleString();
        label.text = `<div style='font-size: 14px; font-weight: 600;'>抵抗値 R: ${value_Ohm_formatted} (${value_kOhm} kΩ)</div>`;
    """)
    widgets['R_slider'].js_on_change('value', r_slider_callback)

    start_measurement_js = CustomJS(args=dict(
        source_clipped=widgets['source_clipped'],
        source_raw=widgets['source_raw'],
        full_data_source_clipped=widgets['full_data_source_clipped'],
        full_data_source_raw=widgets['full_data_source_raw'],
        prefix_select=widgets['prefix_select'],
        number_select=widgets['number_select'],
        asterisk_checkbox=widgets['asterisk_checkbox'],
        r_slider=widgets['R_slider'],
        start_button=widgets['start_button'],
        download_button=widgets['download_button'],
        groups_config=groups_config
    ), code="""
        if (window.animationInterval) { clearInterval(window.animationInterval); }

        download_button.disabled = true;
        start_button.disabled = true;
        source_clipped.data = {'t': [], 'v1': [], 'v2': [], 'v3': []};
        source_raw.data = {'t': [], 'v1': [], 'v2': [], 'v3': []};
        source_clipped.change.emit();
        source_raw.change.emit();

        const prefix = prefix_select.value;
        const number = number_select.value;
        const is_asterisk = asterisk_checkbox.active.includes(0);
        let group = prefix + number;
        if (is_asterisk) { group += '*'; }

        const V0 = 5.0;
        const R = r_slider.value;
        const config = groups_config[group];
        const nominal_caps_uF = config.nominal_caps_uF;
        const tolerance = config.tolerance;
        const noise_level = config.noise_level;

        const true_caps_F = [];
        for (let i = 0; i < nominal_caps_uF.length; i++) {
            const random_error = Math.random() * (tolerance * 2) - tolerance;
            const true_cap_F = nominal_caps_uF[i] * 1e-6 * (1 + random_error);
            true_caps_F.push(true_cap_F);
        }

        const taus = true_caps_F.map(C => R * C);
        const max_tau = Math.max(...taus);
        const t_end = 10 * max_tau;
        const num_data_points = 1000;
        const t_full = Array.from({length: num_data_points}, (_, i) => i * t_end / (num_data_points - 1));

        const voltages_raw = [[], [], []];
        const voltages_clipped = [[], [], []];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < num_data_points; j++) {
                const v_ideal = V0 * (1 - Math.exp(-t_full[j] / taus[i]));
                const u1 = Math.random(); const u2 = Math.random();
                const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                const noise = z0 * (V0 * noise_level);

                const raw_v = v_ideal + noise;
                voltages_raw[i].push(raw_v);

                let clipped_v = raw_v;
                if (clipped_v >= V0) {
                    clipped_v = V0 * 0.99999;
                }
                voltages_clipped[i].push(clipped_v);
            }
        }

        full_data_source_raw.data = {'t': t_full, 'v1': voltages_raw[0], 'v2': voltages_raw[1], 'v3': voltages_raw[2]};
        full_data_source_clipped.data = {'t': t_full, 'v1': voltages_clipped[0], 'v2': voltages_clipped[1], 'v3': voltages_clipped[2]};

        let animation_step = 0;
        const total_steps = 125;
        window.animationInterval = setInterval(() => {
            animation_step++;
            const current_points = Math.floor((animation_step / total_steps) * num_data_points);

            if (current_points >= num_data_points) {
                source_clipped.data = full_data_source_clipped.data;
                source_raw.data = full_data_source_raw.data;
                clearInterval(window.animationInterval);
                download_button.disabled = false;
            } else {
                source_clipped.data = {
                    't': full_data_source_clipped.data.t.slice(0, current_points),
                    'v1': full_data_source_clipped.data.v1.slice(0, current_points),
                    'v2': full_data_source_clipped.data.v2.slice(0, current_points),
                    'v3': full_data_source_clipped.data.v3.slice(0, current_points)
                };
                 source_raw.data = {
                    't': full_data_source_raw.data.t.slice(0, current_points),
                    'v1': full_data_source_raw.data.v1.slice(0, current_points),
                    'v2': full_data_source_raw.data.v2.slice(0, current_points),
                    'v3': full_data_source_raw.data.v3.slice(0, current_points)
                };
            }
            source_clipped.change.emit();
            source_raw.change.emit();
        }, 40);

        start_button.disabled = false;
    """)
    widgets['start_button'].js_on_click(start_measurement_js)

    download_callback = CustomJS(args=dict(
        full_data_source_clipped=widgets['full_data_source_clipped'],
        full_data_source_raw=widgets['full_data_source_raw'],
        prefix_select=widgets['prefix_select'],
        number_select=widgets['number_select'],
        asterisk_checkbox=widgets['asterisk_checkbox'],
        r_slider=widgets['R_slider']
    ), code="""
        const prefix = prefix_select.value;
        const number = number_select.value;
        const is_asterisk = asterisk_checkbox.active.includes(0);
        let group = prefix + number;
        if (is_asterisk) { group += '*'; }

        const V0 = 5.0;
        const R = r_slider.value;
        const data_clipped = full_data_source_clipped.data;
        const data_raw = full_data_source_raw.data;
        const t = data_raw['t'];

        if (t.length === 0) {
            alert("先に計測を開始してください。");
            return;
        }

        let csv_content = "\\uFEFF" + "電源電圧,抵抗値,時刻,V_1_raw(t),V_2_raw(t),V_3_raw(t),V_1_clipped(t),V_2_clipped(t),V_3_clipped(t)\\n";
        for (let i = 0; i < t.length; i++) {
            const row_start = (i === 0) ? `${V0},${R},` : `,,`;
            const row_data = `${t[i]},${data_raw['v1'][i]},${data_raw['v2'][i]},${data_raw['v3'][i]},${data_clipped['v1'][i]},${data_clipped['v2'][i]},${data_clipped['v3'][i]}\\n`;
            csv_content += row_start + row_data;
        }

        const blob = new Blob([csv_content], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        const filename = `measurement_data.csv`;
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    """)
    widgets['download_button'].js_on_click(download_callback)


def create_layout(widgets):
    """
    Creates the final layout by arranging the widgets.
    """
    asterisk_layout = column(widgets['asterisk_label'], widgets['asterisk_checkbox'])

    group_selection_layout = row(
        widgets['prefix_select'],
        widgets['number_select'],
        asterisk_layout,
        sizing_mode="scale_width"
    )

    controls = column(
        group_selection_layout,
        widgets['v0_display'],
        widgets['r_label'],
        widgets['R_slider'],
        widgets['start_button'],
        widgets['download_button'],
        styles={'font-size': '14px'}
    )
    layout = row(controls, widgets['plot'])
    return layout


def save_configs(groups_config, path):
    """
    Saves the group assignment configurations to a CSV file.
    """
    print("\n--- 全グループの割り当て設定 ---")
    csv_output = ["グループ名,公称値1 [uF],公称値2 [uF],公称値3 [uF],精度 [%]\n"]

    for group_name, config in groups_config.items():
        nominal_caps = config['nominal_caps_uF']
        tolerance_percent = int(config['tolerance'] * 100)
        print(f"グループ {group_name:<4}: 公称値 = {str(nominal_caps):<15} | 精度 = ±{tolerance_percent}%")
        csv_row = f"{group_name},{nominal_caps[0]},{nominal_caps[1]},{nominal_caps[2]},{tolerance_percent}\n"
        csv_output.append(csv_row)

    with open(path, 'w', encoding='utf-8-sig') as f:
        f.writelines(csv_output)
    print(f"\n✅ グループ設定を {path} に保存しました。")


def main():
    """
    Main execution function.
    """
    groups_config, all_groups = create_groups_config()
    widgets = create_widgets(all_groups)
    create_callbacks(widgets, groups_config)
    layout = create_layout(widgets)

    save_configs(groups_config, "configs.csv")

    html = file_html(layout, CDN, "CR回路シミュレータ")
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("✅ 'index.html' が生成されました。")


if __name__ == "__main__":
    main()
