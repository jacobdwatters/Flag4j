package com.flag4j.linalg.solvers;

import com.flag4j.Matrix;
import com.flag4j.Vector;
import com.flag4j.linalg.solvers.MatrixLstsqSolver;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixLstsqSolverTests {
    static double[][] aEntries;
    static double[] bEntries;
    static double[] expEntries;

    static Matrix A;
    static Vector b;
    static Vector exp;

    private static void setMatrices() {
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);
    }


    @Test
    void solveTest() {
        MatrixLstsqSolver solver = new MatrixLstsqSolver();

        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{
                {1, 2, 1.55},
                {1, 2.3, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-54.87309197651661, 27.117873450750153, 1.3144161774298762};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-0.5953239804326732, -3.745207845180096, 1.2152364890706673};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));
    }
}
