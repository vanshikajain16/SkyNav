"""Dataset-agnostic loaders for visual-inertial sequences (TUM VI now, EuRoC later).

Everything downstream sees the same containers: ImuData, CamData, GtData, CamCalib,
and a camera model with undistort_to_normalized / project. Timestamps are int64 ns.
"""
from __future__ import annotations                                   # allow forward type hints
from dataclasses import dataclass                                    # plain containers
from pathlib import Path                                             # Windows-safe paths
import numpy as np                                                   # arrays
import cv2                                                           # image IO and camera models
import yaml                                                          # calibration files
from scipy.spatial.transform import Rotation, Slerp                  # rotation interpolation

PROJECT_ROOT = Path(__file__).resolve().parents[1]                   # SkyNav folder


@dataclass                                                           # one dataset on disk
class DatasetSpec:
    name: str                                                        # registry key
    root: Path                                                       # sequence folder
    cam_dir: str                                                     # folder with the PNG frames
    cam_csv: str                                                     # camera index csv
    imu_csv: str                                                     # imu csv
    gt_csv: str                                                      # ground truth csv
    calib_file: str                                                  # calibration yaml
    calib_format: str                                                # "kalibr_camchain" | "euroc_sensor_yaml"
    imu_cols: tuple                                                  # gyro xyz then accel xyz header prefixes
    gt_cols: tuple                                                   # px,py,pz,qw,qx,qy,qz header prefixes
    imu_noise_file: str = ""                                         # yaml with noise densities ("" if none)
    gray_shift: int = 8                                              # 16->8 bit right shift


REGISTRY = {                                                         # known datasets
    "tumvi_room1": DatasetSpec(                                      # TUM VI room1, 512x512, EuRoC/DSO export
        name="tumvi_room1",                                          # key
        root=PROJECT_ROOT / "data" / "tumvi" / "room1",              # extracted folder
        cam_dir="mav0/cam0/data",                                    # frames
        cam_csv="mav0/cam0/data.csv",                                # camera index
        imu_csv="mav0/imu0/data.csv",                                # imu samples
        gt_csv="dso/gt_imu.csv",                                     # IMU pose in mocap world
        calib_file="dso/camchain.yaml",                              # Kalibr calibration
        calib_format="kalibr_camchain",                              # format key
        imu_cols=("w_RS_S_x", "w_RS_S_y", "w_RS_S_z",
                  "a_RS_S_x", "a_RS_S_y", "a_RS_S_z"),               # header prefixes
        gt_cols=("tx", "ty", "tz", "qw", "qx", "qy", "qz"),          # header prefixes
        imu_noise_file="dso/imu_config.yaml",                        # inflated noise values
    ),
}


REGISTRY["tumvi_room4"] = DatasetSpec(                              # TUM VI room4: same export layout as room1
    **{**REGISTRY["tumvi_room1"].__dict__, "name": "tumvi_room4", "root": PROJECT_ROOT / "data" / "tumvi" / "room4"})   # only name and folder differ


@dataclass                                                           # IMU samples
class ImuData:
    t_ns: np.ndarray                                                 # (N,) int64 nanoseconds
    gyro: np.ndarray                                                 # (N,3) rad/s
    acc: np.ndarray                                                  # (N,3) m/s^2


@dataclass                                                           # camera frame index
class CamData:
    t_ns: np.ndarray                                                 # (N,) int64 nanoseconds
    files: list                                                      # list[Path] of PNG frames


@dataclass                                                           # ground truth poses
class GtData:
    t_ns: np.ndarray                                                 # (N,) int64 nanoseconds
    p: np.ndarray                                                    # (N,3) position
    q_wxyz: np.ndarray                                               # (N,4) quaternion, w first

    def interp(self, t_ns):                                          # pose at arbitrary times
        t = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))          # query times
        if t.min() < self.t_ns[0] or t.max() > self.t_ns[-1]:        # outside the GT span
            raise ValueError("query time outside ground-truth span")  # fail loudly
        t0 = self.t_ns[0]                                            # reference time
        ts = (self.t_ns - t0).astype(np.float64) * 1e-9              # GT times in seconds
        tq = (t - t0).astype(np.float64) * 1e-9                      # query times in seconds
        p = np.stack([np.interp(tq, ts, self.p[:, k]) for k in range(3)], axis=1)  # lerp position
        rot = Rotation.from_quat(self.q_wxyz[:, [1, 2, 3, 0]])       # scipy wants x,y,z,w
        return p, Slerp(ts, rot)(tq)                                 # positions and slerped rotations


@dataclass                                                           # camera calibration
class CamCalib:
    K: np.ndarray                                                    # (3,3) intrinsics
    dist: np.ndarray                                                 # (4,) distortion coefficients
    model: str                                                       # "equidistant" | "radtan"
    size: tuple                                                      # (width, height)
    T_cam_imu: np.ndarray                                            # (4,4) maps IMU-frame points to camera frame


def _header(path: Path) -> list:                                     # column names from first line
    with open(path, "r", encoding="utf-8") as f:                     # open the csv
        line = f.readline()                                          # header line
    return [c.strip() for c in line.lstrip("#").strip().split(",")]  # clean names


def _find_cols(header: list, prefixes: tuple) -> list:               # locate columns by header prefix
    idx = []                                                         # result indices
    for pre in prefixes:                                             # one per wanted column
        hit = [i for i, h in enumerate(header) if h.split(" ")[0] == pre]  # exact name before units
        if len(hit) != 1:                                            # must be unique
            raise KeyError(f"column {pre!r} not found uniquely in {header}")  # clear message
        idx.append(hit[0])                                           # keep the index
    return idx                                                       # list of ints


def _load_numeric(path: Path, prefixes: tuple):                      # int64 time + float columns
    header = _header(path)                                           # column names
    cols = _find_cols(header, prefixes)                              # wanted columns
    t = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)    # exact ns timestamps
    v = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, dtype=np.float64)  # float columns
    return t, v                                                      # arrays


def load_imu(spec: DatasetSpec) -> ImuData:                          # IMU samples
    t, v = _load_numeric(spec.root / spec.imu_csv, spec.imu_cols)    # read csv
    return ImuData(t_ns=t, gyro=v[:, 0:3], acc=v[:, 3:6])            # split gyro and accel


def load_gt(spec: DatasetSpec) -> GtData:                            # ground truth
    t, v = _load_numeric(spec.root / spec.gt_csv, spec.gt_cols)      # read csv
    return GtData(t_ns=t, p=v[:, 0:3], q_wxyz=v[:, 3:7])             # position and quaternion


def load_cam(spec: DatasetSpec) -> CamData:                          # camera index
    path = spec.root / spec.cam_csv                                  # csv path
    t = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)    # timestamps
    names = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=str)     # file names
    files = [spec.root / spec.cam_dir / str(n) for n in np.atleast_1d(names)]     # full paths
    if not files[0].is_file():                                       # quick existence check
        raise FileNotFoundError(files[0])                            # fail early
    return CamData(t_ns=t, files=files)                              # container


def _mat(rows) -> np.ndarray:                                        # list of rows -> (4,4) array
    return np.array(rows, dtype=np.float64).reshape(4, 4)            # explicit shape check


def load_calib(spec: DatasetSpec) -> CamCalib:                       # camera calibration
    with open(spec.root / spec.calib_file, "r", encoding="utf-8") as f:   # open yaml
        y = yaml.safe_load(f)                                        # parse
    if spec.calib_format == "kalibr_camchain":                       # TUM VI style
        c = y["cam0"]                                                # first camera
        fx, fy, cx, cy = c["intrinsics"]                             # order verified in tar
        model = {"equidistant": "equidistant", "radtan": "radtan",
                 "radial-tangential": "radtan"}[c["distortion_model"]]   # normalize model name
        return CamCalib(                                             # build calibration
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),  # intrinsic matrix
            dist=np.array(c["distortion_coeffs"], dtype=np.float64),  # distortion
            model=model,                                             # model name
            size=tuple(c["resolution"]),                             # (w, h)
            T_cam_imu=_mat(c["T_cam_imu"]),                          # extrinsic
        )
    if spec.calib_format == "euroc_sensor_yaml":                     # EuRoC style (untested here)
        fx, fy, cx, cy = y["intrinsics"]                             # intrinsics
        T_bs_cam = _mat(y["T_BS"]["data"])                           # sensor in body frame
        imu_yaml = spec.root / "imu0" / "sensor.yaml"                # IMU extrinsic if available
        T_bs_imu = np.eye(4)                                         # body frame equals IMU frame
        if imu_yaml.is_file():                                       # read it when present
            with open(imu_yaml, "r", encoding="utf-8") as f:         # open
                T_bs_imu = _mat(yaml.safe_load(f)["T_BS"]["data"])   # parse
        return CamCalib(                                             # build calibration
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),  # intrinsic matrix
            dist=np.array(y["distortion_coefficients"], dtype=np.float64),  # distortion
            model="radtan",                                          # EuRoC is radial-tangential
            size=tuple(y["resolution"]),                             # (w, h)
            T_cam_imu=np.linalg.inv(T_bs_cam) @ T_bs_imu,            # IMU -> camera
        )
    raise ValueError(f"unknown calib_format {spec.calib_format}")    # unsupported format


def load_imu_noise(spec: DatasetSpec) -> dict:                       # IMU noise parameters
    with open(spec.root / spec.imu_noise_file, "r", encoding="utf-8") as f:   # open yaml
        y = yaml.safe_load(f)                                        # parse (comment lines ignored)
    return {                                                         # canonical keys
        "acc_nd": float(y["accelerometer_noise_density"]),           # m/s^1.5
        "acc_rw": float(y["accelerometer_random_walk"]),             # m/s^2.5
        "gyr_nd": float(y["gyroscope_noise_density"]),               # rad/s^0.5
        "gyr_rw": float(y["gyroscope_random_walk"]),                 # rad/s^1.5
        "rate_hz": float(y["update_rate"]),                          # imu rate
    }


def to_uint8(img: np.ndarray, shift: int = 8) -> np.ndarray:         # 16-bit -> 8-bit
    if img.dtype == np.uint8:                                        # already 8-bit
        return img                                                   # nothing to do
    return (img.astype(np.uint16) >> shift).astype(np.uint8)         # shift right, keep range 0..255


def rate_hz(t_ns: np.ndarray) -> float:                              # median-based rate
    return 1e9 / float(np.median(np.diff(t_ns)))                     # Hz from median period


class RadTan:                                                        # pinhole + radial-tangential
    def __init__(self, calib: CamCalib):                             # keep calibration
        self.calib = calib                                           # store
        self.K = calib.K                                             # intrinsics
        self.dist = calib.dist                                       # distortion

    def undistort_to_normalized(self, px: np.ndarray) -> np.ndarray:  # pixels -> normalized rays
        p = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)       # OpenCV layout
        out = cv2.undistortPoints(p, self.K, self.dist)              # P omitted -> normalized coords
        return out.reshape(-1, 2)                                    # (N,2)

    def project(self, X: np.ndarray) -> np.ndarray:                  # 3D camera points -> pixels
        z = np.zeros(3)                                              # identity pose
        out, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), z, z, self.K, self.dist)  # project
        return out.reshape(-1, 2)                                    # (N,2)

    def undistort_image(self, img: np.ndarray, balance: float = 0.6):  # whole-image undistortion
        return cv2.undistort(img, self.K, self.dist)                 # standard undistortion


class Equidistant:                                                   # pinhole + equidistant (fisheye)
    def __init__(self, calib: CamCalib):                             # keep calibration
        self.calib = calib                                           # store
        self.K = calib.K                                             # intrinsics
        self.dist = calib.dist.reshape(4, 1)                         # fisheye wants (4,1)

    def undistort_to_normalized(self, px: np.ndarray) -> np.ndarray:  # pixels -> normalized rays
        p = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)       # OpenCV layout
        out = cv2.fisheye.undistortPoints(p, self.K, self.dist)      # P omitted -> normalized coords
        return out.reshape(-1, 2)                                    # (N,2)

    def project(self, X: np.ndarray) -> np.ndarray:                  # 3D camera points -> pixels
        z = np.zeros((3, 1))                                         # identity rotation and translation
        out, _ = cv2.fisheye.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), z, z, self.K, self.dist)  # project
        return out.reshape(-1, 2)                                    # (N,2)

    def undistort_image(self, img: np.ndarray, balance: float = 0.6):  # whole-image undistortion
        size = (img.shape[1], img.shape[0])                          # (w, h)
        knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.dist, size, np.eye(3), balance=balance)     # output intrinsics
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.dist, np.eye(3), knew, size, cv2.CV_16SC2)  # remap tables
        return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)  # undistorted image


def make_camera(calib: CamCalib):                                    # config picks the model
    if calib.model == "equidistant":                                 # fisheye
        return Equidistant(calib)                                    # fisheye wrapper
    if calib.model == "radtan":                                      # standard pinhole
        return RadTan(calib)                                         # standard wrapper
    raise ValueError(f"unknown camera model {calib.model}")          # unsupported


class Sequence:                                                      # everything for one sequence
    def __init__(self, name: str):                                   # load by registry name
        self.spec = REGISTRY[name]                                   # dataset spec
        self.imu = load_imu(self.spec)                               # IMU
        self.cam = load_cam(self.spec)                               # camera index
        self.gt = load_gt(self.spec)                                 # ground truth
        self.calib = load_calib(self.spec)                           # calibration
        self.camera = make_camera(self.calib)                        # camera model
        self.noise = load_imu_noise(self.spec) if self.spec.imu_noise_file else None  # IMU noise

    def __len__(self) -> int:                                        # number of frames
        return len(self.cam.t_ns)                                    # frame count

    def frame_raw(self, i: int) -> np.ndarray:                       # untouched 16-bit frame
        img = cv2.imread(str(self.cam.files[i]), cv2.IMREAD_UNCHANGED)  # keep 16-bit
        if img is None:                                              # unreadable file
            raise IOError(f"cannot read {self.cam.files[i]}")        # fail loudly
        return img                                                   # array

    def frame_u8(self, i: int) -> np.ndarray:                        # 8-bit frame for tracking
        return to_uint8(self.frame_raw(i), self.spec.gray_shift)     # convert

    def T_w_c_gt(self, t_ns) -> np.ndarray:                          # GT camera poses, (N,4,4)
        t_ns = np.clip(np.asarray(t_ns, dtype=np.int64), self.gt.t_ns[0], self.gt.t_ns[-1])   # edge frames use the nearest GT pose
        p, R = self.gt.interp(t_ns)                                  # IMU pose in world
        n = len(np.atleast_1d(t_ns))                                 # count
        T_w_i = np.tile(np.eye(4), (n, 1, 1))                        # allocate transforms
        T_w_i[:, :3, :3] = R.as_matrix().reshape(n, 3, 3)            # rotations
        T_w_i[:, :3, 3] = p.reshape(n, 3)                            # translations
        return T_w_i @ np.linalg.inv(self.calib.T_cam_imu)           # T_w_c = T_w_i inv(T_cam_imu)
