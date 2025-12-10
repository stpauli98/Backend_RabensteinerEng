"""Pydantic sheme za validaciju API zahtjeva"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum


class ModelMode(str, Enum):
    """Podržani tipovi ML modela"""
    LINEAR = "LIN"
    DENSE = "Dense"
    CNN = "CNN"
    LSTM = "LSTM"
    AR_LSTM = "AR LSTM"
    SVR_DIR = "SVR_dir"
    SVR_MIMO = "SVR_MIMO"


class ActivationFunction(str, Enum):
    """Podržane aktivacijske funkcije za neuronske mreže"""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    ELU = "elu"
    SELU = "selu"
    SOFTMAX = "softmax"


class SVRKernel(str, Enum):
    """Podržani kerneli za SVR modele"""
    LINEAR = "linear"
    POLY = "poly"
    RBF = "rbf"
    SIGMOID = "sigmoid"


class ModelParameters(BaseModel):
    """
    Shema za validaciju model_parameters u train-models zahtjevu.

    Različiti modeli zahtijevaju različite parametre:
    - Dense/CNN/LSTM/AR_LSTM: LAY, N, EP, ACTF (+ K za CNN)
    - SVR_dir/SVR_MIMO: KERNEL, C, EPSILON
    - LIN: nema dodatnih parametara
    """
    MODE: ModelMode = Field(
        default=ModelMode.DENSE,
        description="Tip ML modela za trening"
    )

    # Parametri za neuronske mreže (Dense, CNN, LSTM, AR LSTM)
    LAY: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Broj skrivenih slojeva (1-50)"
    )
    N: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Broj neurona po sloju (1-2048)"
    )
    EP: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description="Broj epoha treninga (1-5000)"
    )
    ACTF: Optional[ActivationFunction] = Field(
        default=None,
        description="Aktivacijska funkcija"
    )

    # Dodatni parametar za CNN
    K: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Veličina kernela za CNN (1-100)"
    )

    # Parametri za SVR modele
    KERNEL: Optional[SVRKernel] = Field(
        default=None,
        description="Kernel funkcija za SVR"
    )
    C: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=1000,
        description="Regularizacijski parametar za SVR (0.001-1000)"
    )
    EPSILON: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Epsilon za SVR (0-10)"
    )

    model_config = {"extra": "ignore"}  # Ignoriraj nepoznata polja

    @field_validator('MODE', mode='before')
    @classmethod
    def validate_mode(cls, v):
        """Prihvati i string vrijednosti za MODE"""
        if isinstance(v, str):
            # Pokušaj mapirati string na enum
            mode_map = {
                'Linear': ModelMode.LINEAR,
                'LIN': ModelMode.LINEAR,
                'Dense': ModelMode.DENSE,
                'CNN': ModelMode.CNN,
                'LSTM': ModelMode.LSTM,
                'AR LSTM': ModelMode.AR_LSTM,
                'AR_LSTM': ModelMode.AR_LSTM,
                'SVR_dir': ModelMode.SVR_DIR,
                'SVR_MIMO': ModelMode.SVR_MIMO,
            }
            if v in mode_map:
                return mode_map[v]
            # Pokušaj direktno kao enum vrijednost
            try:
                return ModelMode(v)
            except ValueError:
                valid_modes = list(mode_map.keys())
                raise ValueError(f"Nevažeći MODE: '{v}'. Dozvoljeni: {valid_modes}")
        return v


class TrainingSplit(BaseModel):
    """
    Shema za validaciju training_split parametara.

    Kontrolira podjelu podataka na training, validation i test setove.
    """
    testPercentage: float = Field(
        default=20.0,
        ge=5.0,
        le=40.0,
        description="Postotak podataka za test set (5-40%)"
    )
    shuffle: bool = Field(
        default=True,
        description="Miješaj podatke prije podjele"
    )
    validationPercentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=30.0,
        description="Postotak podataka za validation set (0-30%)"
    )

    model_config = {"extra": "ignore"}

    @field_validator('testPercentage', 'validationPercentage', mode='before')
    @classmethod
    def convert_to_float(cls, v):
        """Prihvati i int vrijednosti"""
        if v is not None:
            return float(v)
        return v


class SessionNameChange(BaseModel):
    """Shema za promjenu imena sesije"""
    sessionId: str = Field(
        min_length=1,
        max_length=100,
        description="ID sesije"
    )
    newName: str = Field(
        min_length=1,
        max_length=255,
        description="Novo ime sesije"
    )

    @field_validator('newName')
    @classmethod
    def validate_name(cls, v):
        """Provjeri da ime ne sadrži opasne karaktere"""
        if '<' in v or '>' in v or '&' in v:
            raise ValueError("Ime ne smije sadržavati HTML karaktere")
        return v.strip()


def validate_training_request(data: dict) -> tuple:
    """
    Validiraj train-models zahtjev i vrati validirane parametre.

    Args:
        data: Request JSON podaci s model_parameters i training_split

    Returns:
        tuple: (model_params_dict, training_split_dict)

    Raises:
        ValueError: Ako parametri nisu validni
    """
    try:
        # Validiraj model parametre
        raw_model_params = data.get('model_parameters', {})
        model_params = ModelParameters(**raw_model_params)

        # Validiraj training split
        raw_training_split = data.get('training_split', {})
        training_split = TrainingSplit(**raw_training_split)

        # Dodatna validacija ovisno o tipu modela
        mode = model_params.MODE

        # Provjeri obavezne parametre za neuronske mreže
        if mode in [ModelMode.DENSE, ModelMode.CNN, ModelMode.LSTM, ModelMode.AR_LSTM]:
            missing = []
            if model_params.LAY is None:
                missing.append('LAY (broj slojeva)')
            if model_params.N is None:
                missing.append('N (broj neurona)')
            if model_params.EP is None:
                missing.append('EP (broj epoha)')

            if missing:
                raise ValueError(f"Model {mode.value} zahtijeva parametre: {', '.join(missing)}")

            # CNN zahtijeva K
            if mode == ModelMode.CNN and model_params.K is None:
                raise ValueError("CNN model zahtijeva parametar K (kernel size)")

        # Provjeri obavezne parametre za SVR
        elif mode in [ModelMode.SVR_DIR, ModelMode.SVR_MIMO]:
            missing = []
            if model_params.KERNEL is None:
                missing.append('KERNEL')
            if model_params.C is None:
                missing.append('C')
            if model_params.EPSILON is None:
                missing.append('EPSILON')

            if missing:
                raise ValueError(f"Model {mode.value} zahtijeva parametre: {', '.join(missing)}")

        # Vrati kao dictionary (za kompatibilnost s postojećim kodom)
        return (
            model_params.model_dump(exclude_none=True),
            training_split.model_dump()
        )

    except Exception as e:
        if "validation error" in str(e).lower():
            # Pydantic validation error - formatiiraj poruku
            raise ValueError(f"Neispravni parametri: {str(e)}")
        raise ValueError(str(e))


def validate_model_params_only(data: dict) -> dict:
    """
    Validiraj samo model parametre (bez training_split).

    Args:
        data: Dictionary s model parametrima

    Returns:
        dict: Validirani parametri

    Raises:
        ValueError: Ako parametri nisu validni
    """
    try:
        model_params = ModelParameters(**data)
        return model_params.model_dump(exclude_none=True)
    except Exception as e:
        raise ValueError(f"Neispravni model parametri: {str(e)}")
