package com.flag4j.linalg.decompositions;

import com.flag4j.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ComplexHessenburgTests {
    String[][] aEntries;
    CMatrix A, Q, H, A_hat;

    ComplexHessenburgDecomposition hess;

    @Test

    void testhessDecompTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new String[][]{
                {"1.55-2i", "0", "i"},
                {"25.66-90.25i", "34.5", "3.4+2i"},
                {"-i", "3.4-2i", "16.67+9.2i"}};
        A = new CMatrix(aEntries);
        hess = new ComplexHessenburgDecomposition(true);
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.H());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // ----------------------- Sub-case 1.1 -----------------------
        hess = new ComplexHessenburgDecomposition();
        hess.decompose(A);

        assertEquals(H, hess.getH());

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new String[][]{
                {"1-2i", "0", "-16i", "6-9i"},
                {"4i", "6", "0", "4+3i"},
                {"22+9i", "0", "0", "1+i"},
                {"6+9i", "-25-4i", "1-i", "-1.2+3i"}};
        A = new CMatrix(aEntries);
        hess = new ComplexHessenburgDecomposition(true);
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).mult(Q.H());

        assertEquals(new CMatrix(A.shape.copy()), A.sub(A_hat).roundToZero());

        // ----------------------- Sub-case 2.1 -----------------------
        hess = new ComplexHessenburgDecomposition();
        hess.decompose(A);

        assertEquals(H, hess.getH());
    }
}
