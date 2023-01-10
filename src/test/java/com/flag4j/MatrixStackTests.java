package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixStackTests {

    double[][] expEntries;
    CNumber[][] expCEntries;

    double[][] entries;
    double[][] entries2;
    CNumber[][] CEntries;

    double[] sparseEntries;
    CNumber[][] sparseCEntries;

    int[] indices;
    int[] rowIndices;
    int[] colIndices;

    Shape sparseShape;

    Matrix realDense;
    Matrix realDense2;
    SparseMatrix reapSparse;
    CMatrix complexDense;
    SparseCMatrix sparseCMatrix;
    Vector realDenseVector;
    SparseVector reapSparseVector;
    CVector complexDenseVector;
    SparseCVector sparseCMatrixVector;

    Matrix exp;
    CMatrix expC;


    @Test
    void realDenseTest() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}, {99.2134, -0.23}};

        expEntries = new double[][]{{1, 2, 3, 4, 5}, {4, 14, -5.12, 5, 4}, {0, 9.34, -0.13, 99.2134, -0.23}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 0));


        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5, -0.53}, {5, 4, 1.3}};

        expEntries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}, {4, 5, -0.53}, {5, 4, 1.3}};
        exp = new Matrix(expEntries);

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertEquals(exp, realDense.stack(realDense2, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        entries2 = new double[][]{{4, 5}, {5, 4}};

        realDense = new Matrix(entries);
        realDense2 = new Matrix(entries2);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(realDense2, 1));
    }


    @Test
    void complexDenseTest() {
        // -----------------------  Sub-case 1 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, -0.98824)}, {new CNumber(575, -1.13)},
                {new CNumber(0, 1.33)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(1, -0.98824)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12), new CNumber(575, -1.13)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13), new CNumber(0, 1.33)}};
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertEquals(expC, realDense.stack(complexDense, 0));

        // -----------------------  Sub-case 2 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 0));

        // -----------------------  Sub-case 3 -----------------------
        entries = new double[][]{{1, 2, 3}, {4, 14, -5.12}, {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}};

        expCEntries = new CNumber[][]{{new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(4), new CNumber(14), new CNumber(-5.12)},
                {new CNumber(0), new CNumber(9.34), new CNumber(-0.13)},
                {new CNumber(1.234, -0.234), new CNumber(), new CNumber(92.44, 14.5)}
        };
        expC = new CMatrix(expCEntries);

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertEquals(expC, realDense.stack(complexDense, 1));

        // -----------------------  Sub-case 4 -----------------------
        entries = new double[][]{{1, 2, 3},
                {4, 14, -5.12},
                {0, 9.34, -0.13}};
        CEntries = new CNumber[][]{{new CNumber(1, 0.44)}, {new CNumber(-0.234, -9.234)}};

        realDense = new Matrix(entries);
        complexDense = new CMatrix(CEntries);

        assertThrows(IllegalArgumentException.class, ()->realDense.stack(complexDense, 1));
    }
}
