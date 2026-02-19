import torch
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile
from scipy.signal import butter, filtfilt, find_peaks, resample
from pathlib import Path


# =============================================
# БЛОК 1: Загрузка и чтение ЭКГ с PhysioNet
# =============================================

class PhysioNetECGLoader:
    """
    Загрузчик реальных ЭКГ-записей из базы ECG-ID Database
    https://physionet.org/content/ecgiddb/1.0.0/

    База содержит записи от 90 человек в формате EDF.
    """

    def __init__(self, data_dir="ecgiddb_data"):
        self.data_dir = data_dir
        self.base_url = "https://physionet.org/files/ecgiddb/1.0.0/"

    def download_sample_record(self, person_id=1, record_id=1):
        """
        Скачивает конкретную запись ЭКГ.
        Файлы лежат по пути: Person_XX/rec_Y.edf
        """
        os.makedirs(self.data_dir, exist_ok=True)

        # Формируем путь к файлу на сервере
        person_folder = f"Person_{person_id:02d}"
        filename = f"rec_{record_id}.edf"
        remote_path = f"{self.base_url}{person_folder}/{filename}"

        local_folder = os.path.join(self.data_dir, person_folder)
        os.makedirs(local_folder, exist_ok=True)
        local_path = os.path.join(local_folder, filename)

        if not os.path.exists(local_path):
            print(f"Скачиваю: {remote_path}")
            try:
                urllib.request.urlretrieve(remote_path, local_path)
                print(f"Сохранено: {local_path}")
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
                print("Попробуем альтернативный метод...")
                self._download_with_wfdb(person_id, record_id)
                return local_path
        else:
            print(f"Файл уже существует: {local_path}")

        return local_path

    def _download_with_wfdb(self, person_id, record_id):
        """Альтернативная загрузка через библиотеку wfdb"""
        try:
            import wfdb
            record_name = f"Person_{person_id:02d}/rec_{record_id}"
            wfdb.dl_database(
                'ecgiddb',
                self.data_dir,
                records=[record_name]
            )
        except ImportError:
            print("Установите wfdb: pip install wfdb")

    def read_edf_file(self, filepath):
        """
        Читает EDF-файл и возвращает сигнал + частоту дискретизации.

        Это именно то, что показано в GitHub-ноутбуке:
        https://github.com/TAUforPython/BioMedAI/blob/main/ECG%20EDF%202%20HEADAT.ipynb
        """
        try:
            import pyedflib

            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file

            print(f"\n--- Информация об EDF-файле ---")
            print(f"Количество каналов: {n_channels}")
            print(f"Длительность записи: {f.file_duration} сек")

            channel_labels = f.getSignalLabels()
            sample_rates = []
            signals = []

            for i in range(n_channels):
                label = channel_labels[i]
                fs = f.getSampleFrequency(i)
                signal = f.readSignal(i)
                sample_rates.append(fs)
                signals.append(signal)
                print(f"  Канал {i}: '{label}', "
                      f"частота = {fs} Гц, "
                      f"отсчётов = {len(signal)}")

            f.close()

            # Берём первый канал (обычно это отведение I ЭКГ)
            ecg_signal = signals[0]
            fs = sample_rates[0]

            return ecg_signal, fs, channel_labels

        except ImportError:
            print("pyedflib не установлен, пробую mne...")
            return self._read_with_mne(filepath)

    def _read_with_mne(self, filepath):
        """Альтернативное чтение через MNE"""
        import mne
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = raw.get_data()
        fs = raw.info['sfreq']
        ch_names = raw.ch_names
        print(f"Каналы: {ch_names}, Частота: {fs} Гц")
        return data[0], fs, ch_names


# =============================================
# БЛОК 2: Предобработка ЭКГ-сигнала
# =============================================

class ECGPreprocessor:
    """
    Предобработка реального ЭКГ:
    - Фильтрация (полосовой фильтр 0.5-40 Гц)
    - Нормализация
    - Выделение R-пиков
    - Сегментация по кардиоциклам
    """

    @staticmethod
    def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
        """Полосовой фильтр Баттерворта для удаления шумов"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Защита от некорректных значений
        high = min(high, 0.99)
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered

    @staticmethod
    def normalize(signal):
        """Нормализация в диапазон [0, 1]"""
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        if sig_max - sig_min < 1e-10:
            return np.zeros_like(signal)
        return (signal - sig_min) / (sig_max - sig_min)

    @staticmethod
    def find_r_peaks(signal, fs, min_distance_sec=0.4):
        """
        Поиск R-пиков (основных пиков ЭКГ).
        Используется для сегментации по кардиоциклам.
        """
        min_distance = int(min_distance_sec * fs)
        # Адаптивный порог — 60% от максимума
        threshold = 0.6 * np.max(signal)
        peaks, properties = find_peaks(
            signal,
            height=threshold,
            distance=min_distance
        )
        return peaks

    @staticmethod
    def extract_cardiac_cycles(signal, r_peaks, fs,
                               pre_r_sec=0.2, post_r_sec=0.55):
        """
        Вырезает отдельные кардиоциклы вокруг R-пиков.
        По умолчанию: 200 мс до R и 550 мс после R ≈ 0.75 сек.
        """
        pre_samples = int(pre_r_sec * fs)
        post_samples = int(post_r_sec * fs)
        cycle_length = pre_samples + post_samples

        cycles = []
        for peak in r_peaks:
            start = peak - pre_samples
            end = peak + post_samples
            if start >= 0 and end <= len(signal):
                cycle = signal[start:end]
                cycles.append(cycle)

        return cycles, cycle_length

    @staticmethod
    def compute_average_cycle(cycles, target_length=200):
        """
        Усредняет все кардиоциклы для получения
        «эталонного» (референсного) цикла.
        Ресемплирует все циклы к единой длине.
        """
        resampled = []
        for cycle in cycles:
            resampled.append(resample(cycle, target_length))

        resampled = np.array(resampled)
        mean_cycle = np.mean(resampled, axis=0)
        std_cycle = np.std(resampled, axis=0)

        return mean_cycle, std_cycle, resampled

    def full_pipeline(self, signal, fs, target_length=200):
        """
        Полный конвейер обработки:
        сырой сигнал → фильтрация → нормализация →
        R-пики → кардиоциклы → усреднение
        """
        # 1. Фильтрация
        filtered = self.bandpass_filter(signal, fs)

        # 2. Нормализация
        normalized = self.normalize(filtered)

        # 3. R-пики
        r_peaks = self.find_r_peaks(normalized, fs)
        print(f"Найдено R-пиков: {len(r_peaks)}")

        if len(r_peaks) < 2:
            print("Слишком мало R-пиков! Проверьте сигнал.")
            return None

        # 4. ЧСС (для информации)
        rr_intervals = np.diff(r_peaks) / fs
        heart_rate = 60.0 / np.mean(rr_intervals)
        print(f"Средняя ЧСС: {heart_rate:.1f} уд/мин")
        print(f"Средний RR-интервал: {np.mean(rr_intervals) * 1000:.0f} мс")

        # 5. Кардиоциклы
        cycles, cycle_len = self.extract_cardiac_cycles(
            normalized, r_peaks, fs
        )
        print(f"Извлечено кардиоциклов: {len(cycles)}")

        # 6. Усреднение
        mean_cycle, std_cycle, all_resampled = \
            self.compute_average_cycle(cycles, target_length)

        return {
            'filtered': filtered,
            'normalized': normalized,
            'r_peaks': r_peaks,
            'heart_rate': heart_rate,
            'cycles': cycles,
            'mean_cycle': mean_cycle,
            'std_cycle': std_cycle,
            'all_cycles_resampled': all_resampled,
            'fs': fs
        }


# =============================================
# БЛОК 3: Нейросеть (Reservoir Computing)
# =============================================

class ReservoirNet(torch.nn.Module):
    """
    Нейросеть на основе Reservoir Computing / NG-RC
    - Input: 5 state volumes
    - Reservoir: случайная проекция + нелинейность
    - Readout: обучаемый линейный слой → 2 выхода (Pperi, Vspt)
    """

    def __init__(self, input_dim=5, reservoir_size=50, output_dim=2):
        super().__init__()
        self.W_in = torch.nn.Parameter(
            torch.randn(input_dim, reservoir_size) * 0.1,
            requires_grad=False
        )
        self.W_res = torch.nn.Parameter(
            torch.randn(reservoir_size, reservoir_size) * 0.05,
            requires_grad=False
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(reservoir_size, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, output_dim)
        )

    def forward(self, x):
        r = torch.tanh(x @ self.W_in + (x @ self.W_in) @ self.W_res)
        return self.readout(r)


# =============================================
# БЛОК 4: Гибридная ODE-модель
# =============================================

class HybridODEFunc(torch.nn.Module):
    def __init__(self, p_tensor, net):
        super().__init__()
        self.p = p_tensor
        self.net = net

    def forward(self, t, u):
        Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u
        p = self.p

        (Elvf, Eao, Evc, Ervf, Epa, Epu,
         Rmt, Rav, Rsys, Rtc, Rpv, Rpul,
         Lmt, Lav, Ltc, Lpv,
         Vdlvf, Vdao, Vdvc, Vdrvf, Vdpa, Vdpu,
         P0lvf, P0rvf, lambdalvf, lambdarvf,
         Espt, V0lvf, V0rvf, P0spt, P0pcd,
         V0spt, V0pcd, lambdaspt, lambdapcd,
         Vdspt, Pth) = p

        # Функция активации сердца (один кардиоцикл ≈ 0.75 с)
        e = torch.exp(-80 * ((t % 0.75) - 0.375) ** 2)

        inp = torch.stack([Vlv, Vao, Vvc, Vrv, Vpa]).unsqueeze(0)
        z = self.net(inp).squeeze(0)
        Pperi, Vspt = z[0], z[1]

        Vlvf = Vlv - Vspt
        Vrvf = Vrv + Vspt

        exp_lv = torch.exp(torch.clamp(lambdalvf * (Vlvf - V0lvf), max=88.0))
        exp_rv = torch.exp(torch.clamp(lambdarvf * (Vrvf - V0rvf), max=88.0))

        Plvf = e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp_lv - 1)
        Prvf = e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp_rv - 1)
        Plv = Plvf + Pperi
        Prv = Prvf + Pperi
        Pao = Eao * (Vao - Vdao)
        Pvc = Evc * (Vvc - Vdvc)
        Ppa = Epa * (Vpa - Vdpa) + Pth
        Ppu = Epu * (Vpu - Vdpu) + Pth

        Qsys = (Pao - Pvc) / Rsys
        Qpul = (Ppa - Ppu) / Rpul

        du = torch.zeros_like(u)
        du[0] = (Ppu - Plv - Qmt * Rmt) / Lmt \
            if (Ppu - Plv > 0 or Qmt > 0) else 0.0
        du[1] = (Plv - Pao - Qav * Rav) / Lav \
            if (Plv - Pao > 0 or Qav > 0) else 0.0
        du[2] = (Pvc - Prv - Qtc * Rtc) / Ltc \
            if (Pvc - Prv > 0 or Qtc > 0) else 0.0
        du[3] = (Prv - Ppa - Qpv * Rpv) / Lpv \
            if (Prv - Ppa > 0 or Qpv > 0) else 0.0

        Qmt = torch.clamp(Qmt, min=0.0)
        Qav = torch.clamp(Qav, min=0.0)
        Qtc = torch.clamp(Qtc, min=0.0)
        Qpv = torch.clamp(Qpv, min=0.0)

        du[4] = Qmt - Qav
        du[5] = Qav - Qsys
        du[6] = Qsys - Qtc
        du[7] = Qtc - Qpv
        du[8] = Qpv - Qpul
        du[9] = Qpul - Qmt

        return du


# =============================================
# БЛОК 5: Сравнение модели с реальным ЭКГ
# =============================================

class ModelECGComparator:
    """
    Сравнивает выход ODE-модели с реальной ЭКГ.

    Идея: давление в левом желудочке (Plv) коррелирует
    с электрической активностью сердца, измеряемой на ЭКГ.
    Мы извлекаем из модели «суррогат» ЭКГ и сравниваем
    его форму с реальным эталонным кардиоциклом.
    """

    @staticmethod
    def extract_model_ecg_surrogate(sol_np, p_np, t_np):
        """
        Извлекает суррогатный ЭКГ-сигнал из решения ODE.
        Используем производную давления в ЛЖ (dPlv/dt)
        как суррогат электрической активности.
        """
        Vlv = sol_np[:, 4]
        Vao = sol_np[:, 5]

        Elvf = p_np[0]
        Eao = p_np[1]
        Vdlvf = p_np[16]
        Vdao = p_np[17]

        # Функция активации
        e = np.exp(-80 * ((t_np % 0.75) - 0.375) ** 2)

        # Упрощённое давление ЛЖ
        Plv_approx = e * Elvf * (Vlv - Vdlvf)

        # Производная ~ ЭКГ-суррогат
        ecg_surrogate = np.gradient(Plv_approx, t_np)

        # Нормализация
        ecg_min = ecg_surrogate.min()
        ecg_max = ecg_surrogate.max()
        if ecg_max - ecg_min > 1e-10:
            ecg_surrogate = (ecg_surrogate - ecg_min) / (ecg_max - ecg_min)

        return ecg_surrogate, Plv_approx

    @staticmethod
    def compute_similarity(model_signal, reference_signal):
        """
        Вычисляет метрики сходства:
        - Корреляция Пирсона
        - RMSE
        - DTW-подобная метрика
        """
        # Привести к одной длине
        if len(model_signal) != len(reference_signal):
            reference_signal = resample(
                reference_signal, len(model_signal)
            )

        # Нормализация обоих
        def norm(s):
            s = s - np.mean(s)
            std = np.std(s)
            return s / std if std > 1e-10 else s

        ms = norm(model_signal)
        rs = norm(reference_signal)

        # Корреляция
        correlation = np.corrcoef(ms, rs)[0, 1]

        # RMSE
        rmse = np.sqrt(np.mean((ms - rs) ** 2))

        return {
            'correlation': correlation,
            'rmse': rmse
        }


# =============================================
# БЛОК 6: Визуализация
# =============================================

def plot_ecg_analysis(ecg_data, model_sol, model_t, p_np):
    """
    Комплексная визуализация: реальное ЭКГ + модель
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        'Интеллектуальный анализатор кардиосигнала\n'
        'Сравнение реального ЭКГ (PhysioNet ECG-ID) '
        'с гибридной ODE-моделью',
        fontsize=14, fontweight='bold'
    )

    # --- 1. Сырой ЭКГ-сигнал ---
    ax = axes[0, 0]
    fs = ecg_data['fs']
    t_ecg = np.arange(len(ecg_data['filtered'])) / fs
    ax.plot(t_ecg, ecg_data['filtered'], 'b-', linewidth=0.5)
    # R-пики
    r_peaks = ecg_data['r_peaks']
    ax.plot(
        r_peaks / fs,
        ecg_data['filtered'][r_peaks],
        'rv', markersize=8, label='R-пики'
    )
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда (мВ)')
    ax.set_title(
        f'Реальное ЭКГ (отфильтрованное), '
        f'ЧСС = {ecg_data["heart_rate"]:.0f} уд/мин'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. Все кардиоциклы + средний ---
    ax = axes[0, 1]
    mean_cycle = ecg_data['mean_cycle']
    std_cycle = ecg_data['std_cycle']
    all_cycles = ecg_data['all_cycles_resampled']
    t_cycle = np.linspace(0, 0.75, len(mean_cycle))

    for i, cycle in enumerate(all_cycles):
        ax.plot(t_cycle, cycle, 'b-', alpha=0.15, linewidth=0.5)
    ax.plot(t_cycle, mean_cycle, 'r-', linewidth=2.5,
            label='Средний цикл')
    ax.fill_between(
        t_cycle,
        mean_cycle - std_cycle,
        mean_cycle + std_cycle,
        alpha=0.2, color='red', label='±1 σ'
    )
    ax.set_xlabel('Время в кардиоцикле (с)')
    ax.set_ylabel('Нормализованная амплитуда')
    ax.set_title(
        f'Кардиоциклы ({len(all_cycles)} шт.) + '
        f'референсный средний цикл'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. Переменные состояния модели ---
    ax = axes[1, 0]
    sol_np = model_sol.detach().cpu().numpy()
    t_np = model_t.numpy()
    labels = ['Qmt', 'Qav', 'Qtc', 'Qpv', 'Vlv',
              'Vao', 'Vvc', 'Vrv', 'Vpa', 'Vpu']
    for i in [4, 5, 7]:  # Объёмы: ЛЖ, аорта, ПЖ
        ax.plot(t_np, sol_np[:, i], linewidth=1.5, label=labels[i])
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Объём (мл)')
    ax.set_title('Переменные модели: объёмы камер сердца')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Потоки модели ---
    ax = axes[1, 1]
    for i in [0, 1, 2, 3]:  # Потоки
        ax.plot(t_np, sol_np[:, i], linewidth=1.5, label=labels[i])
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Поток (мл/с)')
    ax.set_title('Переменные модели: потоки через клапаны')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 5. Суррогат ЭКГ из модели ---
    ax = axes[2, 0]
    comparator = ModelECGComparator()
    ecg_surr, Plv = comparator.extract_model_ecg_surrogate(
        sol_np, p_np, t_np
    )

    ax.plot(t_np, ecg_surr, 'g-', linewidth=1.5,
            label='Суррогат ЭКГ (dPlv/dt)')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Нормализованная амплитуда')
    ax.set_title('ЭКГ-суррогат из ODE-модели')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 6. Сравнение форм ---
    ax = axes[2, 1]
    # Ресемплируем средний реальный цикл к длине суррогата модели
    ref_resampled = resample(mean_cycle, len(ecg_surr))

    # Нормализуем оба
    def norm01(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    ax.plot(t_np, norm01(ecg_surr), 'g-', linewidth=2,
            label='Модель (суррогат)')
    ax.plot(t_np, norm01(ref_resampled), 'r--', linewidth=2,
            label='Реальное ЭКГ (средний цикл)')

    metrics = comparator.compute_similarity(ecg_surr, mean_cycle)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Нормализованная амплитуда')
    ax.set_title(
        f'Сравнение форм | '
        f'Корреляция = {metrics["correlation"]:.3f}, '
        f'RMSE = {metrics["rmse"]:.3f}'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cardiac_analysis_result.png', dpi=150,
                bbox_inches='tight')
    plt.show()

    return metrics


# =============================================
# БЛОК 7: Главная программа
# =============================================

def main():
    print("=" * 60)
    print("  ИНТЕЛЛЕКТУАЛЬНЫЙ АНАЛИЗАТОР КАРДИОСИГНАЛА")
    print("  Гибридная ODE-модель + реальное ЭКГ (PhysioNet)")
    print("=" * 60)

    # ----- Шаг 1: Загрузка реального ЭКГ -----
    print("\n[1/5] Загрузка ЭКГ с PhysioNet ECG-ID Database...")

    loader = PhysioNetECGLoader()

    try:
        person_id = int(input(
            "Номер пациента (1-90, Enter=1): "
        ) or "1")
        record_id = int(input(
            "Номер записи (1-20, Enter=1): "
        ) or "1")
    except ValueError:
        person_id, record_id = 1, 1

    edf_path = loader.download_sample_record(person_id, record_id)

    if os.path.exists(edf_path):
        ecg_raw, fs, ch_labels = loader.read_edf_file(edf_path)
        print(f"\nСигнал загружен: {len(ecg_raw)} отсчётов, "
              f"{fs} Гц, {len(ecg_raw) / fs:.1f} сек")
    else:
        print("\nФайл не найден. Генерирую синтетическое ЭКГ...")
        fs = 500
        ecg_raw = _generate_synthetic_ecg(fs, duration=10)

    # ----- Шаг 2: Обработка ЭКГ -----
    print("\n[2/5] Предобработка ЭКГ-сигнала...")
    preprocessor = ECGPreprocessor()
    ecg_data = preprocessor.full_pipeline(ecg_raw, fs, target_length=200)

    if ecg_data is None:
        print("Ошибка обработки ЭКГ. Выход.")
        return

    # ----- Шаг 3: Параметры ODE-модели -----
    print("\n[3/5] Инициализация гибридной ODE-модели...")

    p_np = np.array([
        2.8798, 0.6913, 0.0059, 0.585, 0.369, 0.0073,
        0.0158, 0.018, 1.0889, 0.0237, 0.0055, 0.1552,
        7.6968e-5, 1.2189e-4, 8.0093e-5, 1.4868e-4,
        0, 0, 0, 0, 0, 0,
        0.1203, 0.2157, 0.033, 0.023,
        48.754, 0, 0, 1.1101, 0.5003,
        2, 200, 0.435, 0.03, 2, -4
    ], dtype=np.float32)
    p_tensor = torch.from_numpy(p_np)

    # Начальные условия
    print("\nВведите начальные условия (10 значений через пробел):")
    print("  [Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu]")
    print("  (Enter — значения по умолчанию)")

    try:
        line = input("> ").strip()
        if line:
            u0_vals = list(map(float, line.split()))
            assert len(u0_vals) == 10
        else:
            raise ValueError
    except:
        u0_vals = [
            245.5813, 0.0, 190.0661, 0.0, 94.6812,
            133.3381, 329.7803, 90.7302, 43.0123, 808.4579
        ]
        print("Используются значения по умолчанию.")

    u0 = torch.tensor(u0_vals, dtype=torch.float32)

    # Длительность ~ один кардиоцикл из реальной ЧСС
    rr_sec = 60.0 / ecg_data['heart_rate']
    print(f"\nДлительность кардиоцикла по ЭКГ: {rr_sec * 1000:.0f} мс")

    try:
        dur_input = input(
            f"Длительность моделирования (Enter={rr_sec:.2f} с): "
        ).strip()
        duration = float(dur_input) if dur_input else rr_sec
    except:
        duration = rr_sec

    t = torch.linspace(0.0, duration, 200, dtype=torch.float32)

    # ----- Шаг 4: Решение ODE -----
    print(f"\n[4/5] Решение ODE (t = 0..{duration:.3f} с)...")

    net = ReservoirNet()
    func = HybridODEFunc(p_tensor, net)

    with torch.no_grad():
        sol = odeint(func, u0, t, method='bosh3')

    print("Решение получено!")

    # ----- Шаг 5: Визуализация и сравнение -----
    print("\n[5/5] Визуализация и сравнение с реальным ЭКГ...")

    metrics = plot_ecg_analysis(ecg_data, sol, t, p_np)

    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 60)
    print(f"  Пациент: Person_{person_id:02d}, запись rec_{record_id}")
    print(f"  ЧСС (реальная):          {ecg_data['heart_rate']:.1f} уд/мин")
    print(f"  Кардиоциклов найдено:     {len(ecg_data['cycles'])}")
    print(f"  Корреляция модель/ЭКГ:    {metrics['correlation']:.4f}")
    print(f"  RMSE модель/ЭКГ:          {metrics['rmse']:.4f}")
    print(f"  График сохранён: cardiac_analysis_result.png")
    print("=" * 60)


def _generate_synthetic_ecg(fs=500, duration=10):
    """
    Генерация синтетического ЭКГ
    (если PhysioNet недоступен)
    """
    t = np.arange(0, duration, 1.0 / fs)
    heart_rate = 72
    period = 60.0 / heart_rate

    ecg = np.zeros_like(t)
    for i, ti in enumerate(t):
        phase = (ti % period) / period
        # P-зубец
        ecg[i] += 0.15 * np.exp(-((phase - 0.1) ** 2) / 0.001)
        # QRS-комплекс
        ecg[i] -= 0.1 * np.exp(-((phase - 0.22) ** 2) / 0.0002)
        ecg[i] += 1.0 * np.exp(-((phase - 0.25) ** 2) / 0.0003)
        ecg[i] -= 0.2 * np.exp(-((phase - 0.28) ** 2) / 0.0002)
        # T-зубец
        ecg[i] += 0.3 * np.exp(-((phase - 0.45) ** 2) / 0.003)

    # Добавляем немного шума
    ecg += np.random.normal(0, 0.02, len(ecg))

    return ecg


if __name__ == '__main__':
    main()
