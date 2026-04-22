import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionRecord:
    """
    Registro imutável de uma única detecção do radar.

    Attributes:
        time:        Tempo de simulação em que a detecção ocorreu (s).
        target_idx:  Índice do target detectado (0-based).
        range_m:     Distância do radar ao target no momento da detecção (m).
        azimuth_deg: Azimute do target no momento da detecção (°, [0, 360)).
        deg_error:   Erro angular em relação ao centro do feixe (°),
                     quantizado por deg_step. 0 = centro exato do feixe.
    """
    time:        float
    target_idx:  int
    range_m:     float
    azimuth_deg: float
    deg_error:   float


class DetectionLog:
    """
    Coleção de DetectionRecord produzidos durante a simulação.

    Responsabilidades:
        - Acumular registros de detecção a cada passo da simulação.
        - Fornecer acesso aos dados para visualização e análise.
        - Exportar os dados para CSV.
    """

    def __init__(self):
        self._records: list[DetectionRecord] = []

    @property
    def records(self) -> list[DetectionRecord]:
        """Lista completa de registros (somente leitura)."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    def by_target(self, target_idx: int) -> list[DetectionRecord]:
        """Retorna todos os registros de um target específico."""
        return [r for r in self._records if r.target_idx == target_idx]

    def target_indices(self) -> list[int]:
        """Retorna os índices únicos de targets detectados, em ordem de primeira detecção."""
        seen: list[int] = []
        for r in self._records:
            if r.target_idx not in seen:
                seen.append(r.target_idx)
        return seen

    def add(self, record: DetectionRecord) -> None:
        """Adiciona um registro de detecção ao log."""
        self._records.append(record)

    def clear(self) -> None:
        """Remove todos os registros (útil para reiniciar a simulação)."""
        self._records.clear()

    def export(self, path: str | Path = "detections.csv") -> Path:
        """
        Salva todos os registros em um arquivo CSV.

        O arquivo é criado (com diretórios pai, se necessário) e sobrescrito
        caso já exista. Campos numéricos são formatados com precisão controlada
        para eliminar artefatos de ponto flutuante.

        Args:
            path: Caminho do arquivo de saída (str ou Path).

        Returns:
            Path resolvido do arquivo criado.
        """
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ['time', 'target_idx', 'range_m', 'azimuth_deg', 'deg_error']

        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                writer.writerow({
                    'time':        f'{rec.time:.6g}',
                    'target_idx':  rec.target_idx,
                    'range_m':     f'{rec.range_m:.4f}',
                    'azimuth_deg': f'{rec.azimuth_deg:.4f}',
                    'deg_error':   f'{rec.deg_error:.4f}',
                })

        return output

