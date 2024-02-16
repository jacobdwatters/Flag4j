package com.flag4j.linalg;

import com.flag4j.dense.CMatrix;
import com.flag4j.dense.Matrix;
import com.flag4j.linalg.decompositions.cholesky.ComplexCholeskyDecomposition;
import com.flag4j.linalg.decompositions.cholesky.RealCholeskyDecomposition;
import com.flag4j.linalg.decompositions.lu.ComplexLUDecomposition;
import com.flag4j.linalg.decompositions.lu.RealLUDecomposition;
import com.flag4j.linalg.decompositions.qr.ComplexQRDecomposition;
import com.flag4j.linalg.decompositions.qr.RealQRDecomposition;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DecomposeTests {

    double[][] aEntries;
    String[][] acEntries;
    Matrix A;
    CMatrix AC;

    @Test
    void luTestCase() {
        RealLUDecomposition lu = new RealLUDecomposition();
        ComplexLUDecomposition lu_complex = new ComplexLUDecomposition();

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[][]
                {{1.4, 5.67, -9.23},
                {899.23, 7.2, 0.0},
                {0.0, 4.77, 71.578}};
        A = new Matrix(aEntries);
        lu.decompose(A);

        Matrix[] res = Decompose.lu(A);
        assertArrayEquals(res, new Matrix[]{lu.getP().toDense(), lu.getL(), lu.getU()});

        // ----------------------- Sub-case 2 -----------------------
        acEntries = new String[][]{
                {"1.255+45.1i", "-99.24+0.024i"},
                {"6.466", "8.4-2.45i"},
                {"1.05-47.1i", "66.2+3.5i"}};
        AC = new CMatrix(acEntries);
        lu_complex.decompose(AC);

        CMatrix[] res_complex = Decompose.lu(AC);
        assertArrayEquals(res_complex, new CMatrix[]{lu_complex.getP().toDense().toComplex(),
                lu_complex.getL(), lu_complex.getU()});
    }


    @Test
    void qrTestCase() {
        RealQRDecomposition qr = new RealQRDecomposition();
        ComplexQRDecomposition qr_complex = new ComplexQRDecomposition();

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[][]{
                {1.0, 5.6, -9.355, 215.0},
                {56.0, 1.0, 15.2, 14.0},
                {2.4, -0.00025, 1.0, 0.0},
                {1.0, 49.4, 106.2, -8.5}};
        A = new Matrix(aEntries);
        qr.decompose(A);

        Matrix[] res = Decompose.qr(A);
        assertArrayEquals(res, new Matrix[]{qr.getQ(), qr.getR()});

        // ----------------------- Sub-case 2 -----------------------
        acEntries = new String[][]{
                {"2.45-8.4i", "34.5i", "-i", "9.35+0.936i"},
                {"-21.1255-4i", "14.0045-0.99835i", "24.5", "48i"},
                {"i", "900.3516+8891.331i", "-0.24+0.00024i", "-2.5 + 14i"}};
        AC = new CMatrix(acEntries);
        qr_complex.decompose(AC);

        CMatrix[] res_complex = Decompose.qr(AC);
        assertArrayEquals(res_complex, new CMatrix[]{qr_complex.getQ(), qr_complex.getR()});
    }


    @Test
    void choleskyTestCase() {
        RealCholeskyDecomposition cholesky = new RealCholeskyDecomposition();
        ComplexCholeskyDecomposition cholesky_complex = new ComplexCholeskyDecomposition();

        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[][]{
                {2.0, -1.0, 0.0},
                {-1.0, 2.0, -1.0},
                {0.0, -1.0, 2.0}};
        A = new Matrix(aEntries);
        cholesky.decompose(A);

        assertEquals(Decompose.cholesky(A), cholesky.decompose(A).getL());

        // ----------------------- Sub-case 2 -----------------------
        acEntries = new String[][]{
                {"1", "-2i"},
                {"2i", "5"}};
        AC = new CMatrix(acEntries);
        cholesky_complex.decompose(AC);

        assertEquals(Decompose.cholesky(AC), cholesky_complex.decompose(AC).getL());
    }
}
