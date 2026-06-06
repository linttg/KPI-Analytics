import numpy as np

class StrategyEngine:

    @staticmethod
    def calculate_ahp(matrix):
        """
        Menghitung bobot prioritas dan Consistency Ratio
        """

        n = matrix.shape[0]

        # Normalisasi matriks
        norm_matrix = matrix / matrix.sum(axis=0)

        # Priority weights
        weights = norm_matrix.mean(axis=1)

        # Consistency Ratio
        lambda_max = np.real(np.linalg.eigvals(matrix).max())

        ci = (lambda_max - n) / (n - 1)

        ri_table = {
            1: 0.00,
            2: 0.00,
            3: 0.58,
            4: 0.90,
            5: 1.12
        }

        ri = ri_table.get(n, 1.12)

        cr = ci / ri if ri != 0 else 0

        return weights, cr

    @staticmethod
    def synthesize_decision(kpi_scores, criteria_weights):
        """
        Menggabungkan skor KPI dan bobot AHP
        """

        final_scores = np.dot(kpi_scores, criteria_weights)

        return final_scores