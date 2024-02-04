package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import com.flag4j.linalg.decompositions.qr.ComplexQRDecompositionOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexQRTests {

    double[][] aEntriesReal;
    String[][] aEntries;
    CMatrix A, Q, R, A_hat;
    ComplexQRDecompositionOld qr;

    @Test
    void fullTestCase() {
        // Tests account for numerical loss of precision.
        qr = new ComplexQRDecompositionOld();

        // --------------------------- Sub-case 1 ---------------------------
        aEntriesReal = new double[][]
                {{0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
                {1, 0, 0, 0}};
        A = new CMatrix(aEntriesReal);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 2 ---------------------------
        aEntriesReal = new double[][]{
                {0, 0, 0},
                {0, 0, -1},
                {0, 1, 0}
        };
        A = new CMatrix(aEntriesReal);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 3 ---------------------------
        aEntries = new String[][]{
                {"2+i", "-i"},
                {"3-2i", "4i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 4 ---------------------------
        aEntries = new String[][]{
                {"2.45-8.4i", "34.5i", "-i"},
                {"-21.1255-4i", "14.0045-0.99835i", "24.5"},
                {"i", "0", "-0.24+0.00024i"},
                {"0", "48i", "-2.5 + 14i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 4 ---------------------------
        aEntries = new String[][]{
                {"2.45-8.4i", "34.5i", "-i", "9.35+0.936i"},
                {"-21.1255-4i", "14.0045-0.99835i", "24.5", "48i"},
                {"i", "900.3516+8891.331i", "-0.24+0.00024i", "-2.5 + 14i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero(1.0E-10));
    }

    @Test
    void reducedTestCase() {
        // Tests account for numerical loss of precision.
        qr = new ComplexQRDecompositionOld(false);

        // --------------------------- Sub-case 1 ---------------------------
        aEntriesReal = new double[][]
                {{0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
                {1, 0, 0, 0}};
        A = new CMatrix(aEntriesReal);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 2 ---------------------------
        aEntriesReal = new double[][]{
                {0, 0, 0},
                {0, 0, -1},
                {0, 1, 0}
        };
        A = new CMatrix(aEntriesReal);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 3 ---------------------------
        aEntries = new String[][]{
                {"2+i", "-i"},
                {"3-2i", "4i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 4 ---------------------------
        aEntries = new String[][]{
                {"2.45-8.4i", "34.5i", "-i"},
                {"-21.1255-4i", "14.0045-0.99835i", "24.5"},
                {"i", "0", "-0.24+0.00024i"},
                {"0", "48i", "-2.5 + 14i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // --------------------------- Sub-case 4 ---------------------------
        aEntries = new String[][]{
                {"2.45-8.4i", "34.5i", "-i", "9.35+0.936i"},
                {"-21.1255-4i", "14.0045-0.99835i", "24.5", "48i"},
                {"i", "900.3516+8891.331i", "-0.24+0.00024i", "-2.5 + 14i"}};
        A = new CMatrix(aEntries);

        qr.decompose(A);
        Q = qr.getQ();
        R = qr.getR();
        A_hat = Q.mult(R);

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero(1.0E-10));
    }
}
