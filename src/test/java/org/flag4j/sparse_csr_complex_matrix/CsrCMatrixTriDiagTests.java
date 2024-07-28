package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixTriDiagTests {

    static CsrCMatrix A;
    static Shape aShape;
    static CNumber[] aEntries;
    static int[] aRowPointers;
    static int[] aColIndices;

    @Test
    void traceTest() {
        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new CNumber[]{new CNumber(0.13392, 0.67581), new CNumber(0.08953, 0.43612),
                new CNumber(0.68598, 0.57467), new CNumber(0.19996, 0.4443), new CNumber(0.1453, 0.59643)};
        aRowPointers = new int[]{0, 1, 1, 2, 2, 5, 5};
        aColIndices = new int[]{5, 0, 0, 1, 3};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new CNumber(0), A.trace());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new CNumber[]{new CNumber(0.90327, 0.45253), new CNumber(0.21721, 0.28695),
                new CNumber(0.65185, 0.93707), new CNumber(0.48592, 0.63105), new CNumber(0.96722, 0.76818)};
        aRowPointers = new int[]{0, 1, 1, 2, 4, 5, 5};
        aColIndices = new int[]{3, 5, 3, 4, 5};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new CNumber(0.65185, 0.93707), A.trace());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new CNumber[]{new CNumber(0.52333, 0.67155), new CNumber(0.79849, 0.05489),
                new CNumber(0.2229, 0.67036), new CNumber(0.65271, 0.51699), new CNumber(0.63722, 0.55373),
                new CNumber(0.21806, 0.5938), new CNumber(0.06624, 0.41699), new CNumber(0.32211, 0.1279),
                new CNumber(0.11324, 0.21277), new CNumber(0.45704, 0.75931), new CNumber(0.13948, 0.53299),
                new CNumber(0.13934, 0.59231), new CNumber(0.30193, 0.98664), new CNumber(0.11591, 0.98686),
                new CNumber(0.29993, 0.8055), new CNumber(0.43436, 0.64936), new CNumber(0.9495, 0.32514),
                new CNumber(0.22636, 0.4559), new CNumber(0.58931, 0.3885), new CNumber(0.332, 0.82381),
                new CNumber(0.11975, 0.11127), new CNumber(0.46906, 0.80406)};
        aRowPointers = new int[]{0, 4, 5, 7, 8, 9, 10, 12, 13, 15, 19, 21, 22};
        aColIndices = new int[]{2, 3, 5, 11, 10, 2, 11, 4, 8, 11, 8, 9, 0, 3, 10, 3, 6, 7, 11, 1, 4, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        assertEquals(new CNumber(0.21806, 0.5938), A.trace());

        // ----------------------- sub-case 4 -----------------------
        assertThrows(LinearAlgebraException.class, ()->new CsrCMatrix(12, 4).tr());
    }


    @Test
    void getDiagTests() {
        CooCVector exp;

        // ----------------------- sub-case 1 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new CNumber[]{new CNumber(0.16488, 0.46447), new CNumber(0.3774, 0.35246),
                new CNumber(0.48798, 0.43226), new CNumber(0.48544, 0.0083), new CNumber(0.77148, 0.92185)};
        aRowPointers = new int[]{0, 0, 0, 1, 3, 5, 5};
        aColIndices = new int[]{2, 0, 5, 2, 4};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.16488, 0.46447),
                new CNumber(0.0, 0.0), new CNumber(0.77148, 0.92185), new CNumber(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 2 -----------------------
        aShape = new Shape(6, 6);
        aEntries = new CNumber[]{new CNumber(0.3583, 0.11616), new CNumber(0.38055, 0.12452),
                new CNumber(0.97075, 0.0994), new CNumber(0.43841, 0.0319), new CNumber(0.78954, 0.72668)};
        aRowPointers = new int[]{0, 1, 3, 4, 4, 4, 5};
        aColIndices = new int[]{4, 0, 3, 5, 2};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(0, 0, 0, 0, 0, 0).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 3 -----------------------
        aShape = new Shape(12, 12);
        aEntries = new CNumber[]{new CNumber(0.13994, 0.01355), new CNumber(0.5696, 0.69936),
                new CNumber(0.1473, 0.92664), new CNumber(0.40113, 0.88561), new CNumber(0.45177, 0.86077),
                new CNumber(0.91248, 0.28304), new CNumber(0.71058, 0.73788), new CNumber(0.04337, 0.79422),
                new CNumber(0.26867, 0.54323), new CNumber(0.28884, 0.58676), new CNumber(0.62439, 0.17585),
                new CNumber(0.76891, 0.03482), new CNumber(0.02111, 0.58864), new CNumber(0.36376, 0.44496)};
        aRowPointers = new int[]{0, 1, 4, 5, 6, 8, 10, 11, 12, 13, 13, 14, 14};
        aColIndices = new int[]{7, 3, 5, 8, 6, 3, 6, 7, 3, 5, 11, 10, 8, 11};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0),
                new CNumber(0.91248, 0.28304), new CNumber(0.0, 0.0), new CNumber(0.28884, 0.58676),
                new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.02111, 0.58864),
                new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 4 -----------------------
        aShape = new Shape(3, 7);
        aEntries = new CNumber[]{new CNumber(0.85218, 0.08775), new CNumber(0.16499, 0.26153), new CNumber(0.36642, 0.63926), new CNumber(0.63494, 0.79019)};
        aRowPointers = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 3, 5, 1};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new CNumber(0.85218, 0.08775), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)).toCoo();
        assertEquals(exp, A.getDiag());

        // ----------------------- sub-case 5 -----------------------
        aShape = new Shape(7, 3);
        aEntries = new CNumber[]{new CNumber(0.38881, 0.35982), new CNumber(0.82873, 0.85744),
                new CNumber(0.22975, 0.7268), new CNumber(0.9345, 0.92117)};
        aRowPointers = new int[]{0, 1, 1, 2, 2, 2, 3, 4};
        aColIndices = new int[]{0, 2, 2, 0};
        A = new CsrCMatrix(aShape, aEntries, aRowPointers, aColIndices);
        exp = new CVector(new CNumber(0.38881, 0.35982), new CNumber(0.0, 0.0),
                new CNumber(0.82873, 0.85744)).toCoo();
        assertEquals(exp, A.getDiag());
    }
}
