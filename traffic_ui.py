"""Giao diện Gradio cho dự báo lưu lượng và báo cáo PeMS04."""

from dataclasses import dataclass
from functools import lru_cache
from html import escape

import gradio as gr
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


HORIZONS = (15, 30, 60)
SCENARIOS = {"clean": "Bình thường", "random": "Mất ngẫu nhiên", "region": "Mất theo vùng"}
ALL = "all"
TIME_FORMAT = "%Y-%m-%d %H:%M"


@dataclass
class ForecastView:
    result: object
    description: str
    csv_path: str


def metric_cards(items):
    cards = []
    for label, text, unit in items:
        cards.append(
            '<div style="flex:1 1 135px;border-left:3px solid #178578;'
            'padding:4px 14px;margin:5px 0">'
            f'<div>{escape(label)}</div><strong style="font-size:1.65rem">{escape(text)}</strong>'
            f' <small>{escape(unit)}</small></div>'
        )
    return '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(cards) + "</div>"


def forecast_plot(result, sensor, position):
    history = result.history[sensor]
    target = result.origin + pd.to_timedelta(HORIZONS, unit="min")
    fig = Figure(figsize=(9, 3.1), tight_layout=True, facecolor="white")
    ax = fig.subplots()
    ax.plot(history.index, history, color="#536579", linewidth=1.8, label="Lịch sử")
    ax.axvline(result.origin, linewidth=1, color="#a7b2ba", linestyle=":")
    ax.plot(target, result.clean[position], "o--", color="#087b6d", label="Dự báo")
    if result.simulated is not None:
        ax.plot(target, result.simulated[position], "s--", color="#ad4b12", label="Khi cảm biến hỏng")
    actual = result.actual[position]
    observed = np.isfinite(actual)
    if observed.any():
        ax.scatter(target[observed], actual[observed], marker="x", s=55,
                   linewidths=2, color="#5134a5", label="Thực tế", zorder=4)
    ax.set_title(f"Cảm biến {sensor} · {result.origin:%d/%m/%Y}", loc="left", fontsize=12)
    ax.set_ylabel("xe/5 phút")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="y", color="#d9e1e5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    return fig


def report_table(raw, horizon):
    shown = pd.DataFrame({
        "Kịch bản": raw["scenario"].map(SCENARIOS),
        "Mất (%)": raw["miss_pct"],
        "Mất (phút)": raw["miss_minutes"],
        "Số cảm biến hỏng": raw["n_sensors_lost"],
        "MAE": raw[f"MAE_{horizon}"],
        "RMSE": raw[f"RMSE_{horizon}"],
        "MAPE (%)": raw[f"MAPE_{horizon}"],
    })
    return shown.reset_index(drop=True)


def build_ui(service):
    sensors = tuple(service.sensor_ids)
    positions = {sensor: i for i, sensor in enumerate(sensors)}
    demo_index = service.valid_times(service.demo)
    demo_times = demo_index.strftime(TIME_FORMAT).tolist()
    morning = demo_index[(demo_index.hour == 8) & (demo_index.minute == 0)]
    demo_origin = (morning[0] if len(morning) else demo_index[0]).strftime(TIME_FORMAT)
    report = service.report
    baseline = report.loc[report["scenario"] == "clean"].iloc[0]

    @lru_cache(maxsize=4)
    def report_csv(scenario):
        subset = report if scenario == ALL else report.loc[report["scenario"] == scenario]
        return str(service.export_report(subset))

    with gr.Blocks(title="Dự báo lưu lượng PeMS04") as app:
        gr.Markdown("# Dự báo lưu lượng giao thông\nSTID · PeMS04 · 307 cảm biến · Dữ liệu mỗi 5 phút")
        uploaded_frame = gr.State(None)
        selected_origin = gr.State(demo_origin)
        last_forecast = gr.State(None)

        with gr.Tab("Dự báo"):
            source = gr.Radio(
                [("PeMS04 có sẵn", "demo"), ("Tải CSV riêng", "csv")],
                value="demo", label="Nguồn dữ liệu",
            )
            with gr.Group(visible=False) as upload_area:
                with gr.Row():
                    csv_file = gr.File(label="Dữ liệu lưu lượng (.csv)", file_types=[".csv"],
                                       type="filepath", height=110, scale=3)
                    gr.DownloadButton("Tải CSV mẫu", value=service.csv_template(), size="sm")
                gr.Markdown("CSV gồm cột `timestamp` và 307 cột cảm biến `0` đến `306`. "
                            "Cần ít nhất 24 mốc liên tiếp, cách nhau 5 phút.")
            upload_status = gr.Textbox(label="Kiểm tra CSV", visible=False, interactive=False)
            with gr.Row():
                origin = gr.Dropdown(demo_times, value=demo_origin, label="Thời điểm dự báo",
                                     min_width=210, scale=2)
                sensor = gr.Dropdown(list(sensors), value=sensors[0], label="Cảm biến",
                                     min_width=130)
                run = gr.Button("Dự báo", variant="primary", min_width=110)

            cards = gr.HTML(visible=False)
            plot = gr.Plot(label="Lưu lượng theo thời gian", visible=False)
            download = gr.DownloadButton("Tải dự báo 307 cảm biến (CSV)", visible=False, size="sm")

            with gr.Accordion("Giả lập cảm biến hỏng", open=False) as simulation_area:
                with gr.Row():
                    mode = gr.Radio([("Ngẫu nhiên", "random"), ("Theo vùng", "region")],
                                    value="random", label="Kiểu hỏng")
                    percent = gr.Radio([("5%", 5), ("20%", 20), ("30%", 30)],
                                       value=5, label="Tỷ lệ cảm biến hỏng")
                    duration = gr.Radio([15, 30, 45], value=15, label="Số phút mất ở cuối cửa sổ")
                region_info = gr.Markdown(
                    "Hỏng 10% cảm biến gần nhau. Cảm biến đang chọn là tâm khi bấm mô phỏng.",
                    visible=False,
                )
                simulate = gr.Button("Chạy mô phỏng")
                comparison = gr.Dataframe(interactive=False, visible=False, label="So sánh dự báo")

        with gr.Tab("Đánh giá mô hình"):
            gr.Markdown("### Kết quả trên tập test sạch\nTrung bình ba mốc dự báo.")
            gr.HTML(metric_cards([
                ("MAE", f"{baseline.MAE_avg:.3f}", "xe/5 phút"),
                ("RMSE", f"{baseline.RMSE_avg:.3f}", "xe/5 phút"),
                ("MAPE", f"{baseline.MAPE_avg:.3f}", "%"),
            ]))
            gr.Markdown("### So sánh 12 kịch bản trên toàn tập test")
            with gr.Row():
                case_filter = gr.Dropdown(
                    [("Tất cả", ALL)] + [(name, key) for key, name in SCENARIOS.items()],
                    value=ALL, label="Kịch bản",
                )
                horizon_filter = gr.Dropdown(
                    [("Trung bình 3 mốc", "avg")] + [(f"Sau {h} phút", str(h)) for h in HORIZONS],
                    value="avg", label="Mốc dự báo",
                )
            benchmark = gr.Dataframe(value=report_table(report, "avg"), interactive=False,
                                     label="Sai số trên toàn tập test", max_height=520)
            report_download = gr.DownloadButton("Tải các kịch bản đã lọc (CSV, đủ 3 mốc)",
                                                value=report_csv(ALL), size="sm")

        result_components = [last_forecast, cards, plot, comparison, download]

        def clear_result():
            return {last_forecast: None, cards: gr.update(visible=False, value=""),
                    plot: gr.update(visible=False, value=None),
                    comparison: gr.update(visible=False, value=None),
                    download: gr.update(visible=False, value=None)}

        def select_data(source_name, path):
            uploaded = source_name == "csv"
            frame, choices, selected, message = None, demo_times, demo_origin, ""
            if uploaded:
                choices, selected = [], None
                message = "Chọn file CSV để dự báo."
                if path:
                    try:
                        frame = service.load_csv(path)
                        choices = service.valid_times(frame).strftime(TIME_FORMAT).tolist()
                        selected = choices[-1]
                        message = f"Hợp lệ · {len(frame)} mốc thời gian · {len(frame.columns)} cảm biến."
                    except (ValueError, OSError) as exc:
                        message = f"CSV không hợp lệ. {exc}"
            return {
                **clear_result(), uploaded_frame: frame, selected_origin: selected,
                upload_area: gr.update(visible=uploaded),
                simulation_area: gr.update(visible=not uploaded, open=False),
                upload_status: gr.update(visible=uploaded, value=message),
                origin: gr.update(choices=choices, value=selected, interactive=bool(choices)),
                run: gr.update(interactive=bool(choices)), simulate: gr.update(interactive=not uploaded),
            }

        def show_forecast(view, sensor_id):
            if view is None:
                return clear_result()
            result = view.result
            i = positions[sensor_id]
            summary = metric_cards([
                (f"Sau {h} phút", f"{val:.1f}", "xe/5 phút") for h, val in zip(HORIZONS, result.clean[i])
            ])
            comparison_view = gr.update(visible=False)
            if result.simulated is not None:
                summary = "<strong>Dự báo trước giả lập</strong>" + summary
                simulated = result.simulated[i]
                table = pd.DataFrame({
                    "Mốc": [f"+{h} phút" for h in HORIZONS],
                    "Bình thường": result.clean[i].round(2),
                    "Khi hỏng": simulated.round(2),
                    "Chênh lệch": (simulated - result.clean[i]).round(2),
                })
                condition = "nằm trong nhóm hỏng" if sensor_id in result.affected_ids else "ngoài nhóm hỏng"
                comparison_view = gr.update(visible=True, value=table,
                                            label=f"{view.description} · Cảm biến {sensor_id} {condition}")
            return {
                cards: gr.update(visible=True, value=summary),
                plot: gr.update(visible=True, value=forecast_plot(result, sensor_id, i)),
                comparison: comparison_view,
                download: gr.update(visible=True, value=view.csv_path),
            }

        def compute(source_name, frame, time, sensor_id, fault="clean", pct=5, mins=15):
            data = service.demo if source_name == "demo" else frame
            if data is None or time is None or sensor_id not in positions:
                gr.Warning("Chọn dữ liệu, thời điểm và cảm biến hợp lệ.")
                return clear_result()
            is_region = fault == "region"
            pct = 10 if is_region else int(pct)
            try:
                result = service.forecast(data, time, mode=fault, percent=pct, minutes=int(mins),
                                          center=sensor_id if is_region else None, seed=2026)
                csv_path = str(service.export_forecast(result))
            except (ValueError, OSError) as exc:
                gr.Warning(str(exc))
                return clear_result()
            description = f"{SCENARIOS[fault]} · {pct}% · {mins} phút"
            if is_region:
                description += f" · Tâm {sensor_id}"
            view = ForecastView(result, description + " · xe/5 phút", csv_path)
            return {last_forecast: view, **show_forecast(view, sensor_id)}

        def show_sensor(view, sensor_id):
            if view is None or sensor_id not in positions:
                return clear_result()
            return show_forecast(view, sensor_id)

        def reset_simulation(view):
            if view is not None and view.result.simulated is not None:
                return clear_result()
            return {c: gr.skip() for c in result_components}

        def select_mode(fault, view):
            region = fault == "region"
            return {
                **reset_simulation(view),
                percent: gr.update(visible=not region),
                duration: gr.update(choices=[15, 30] if region else [15, 30, 45], value=15),
                region_info: gr.update(visible=region),
            }

        def filter_report(scenario, horizon):
            subset = report if scenario == ALL else report.loc[report["scenario"] == scenario]
            return report_table(subset, horizon), report_csv(scenario)

        def select_time(time):
            return {**clear_result(), selected_origin: time}

        # Keep source, result and export ordered when a user changes controls rapidly.
        ordered = {"concurrency_id": "forecast-controls", "concurrency_limit": 1,
                   "show_progress": "minimal"}
        gr.on([source.input, csv_file.change], select_data, [source, csv_file],
              [uploaded_frame, selected_origin, upload_area, simulation_area, upload_status, origin, run, simulate,
               *result_components],
              api_name="select_data", **ordered)
        origin.input(select_time, origin, [selected_origin, *result_components],
                     api_name="select_time", **ordered)
        forecast_inputs = [source, uploaded_frame, selected_origin, sensor]
        run.click(compute, forecast_inputs, result_components, api_name="run_forecast", **ordered)
        simulate.click(compute, forecast_inputs + [mode, percent, duration], result_components,
                       api_name="run_simulation", **ordered)
        sensor.input(show_sensor, [last_forecast, sensor], result_components,
                     api_name="view_sensor", **ordered)
        mode.input(select_mode, [mode, last_forecast],
                   [percent, duration, region_info, *result_components], api_name="select_fault", **ordered)
        gr.on([percent.input, duration.input], reset_simulation, [last_forecast], result_components,
              api_name="reset_simulation", **ordered)
        gr.on([case_filter.input, horizon_filter.input], filter_report, [case_filter, horizon_filter],
              [benchmark, report_download], api_name="filter_report", show_progress="hidden")
    return app
