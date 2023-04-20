package com.flag4j.linalg.solvers;

import com.flag4j.Matrix;
import com.flag4j.Vector;
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
        expEntries = new double[]{-54.873091976516804, 27.117873450750242, 1.3144161774298757};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-0.5953239804326801, -3.7452078451800963, 1.2152364890706675};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));
    }
}
