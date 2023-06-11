package com.flag4j.linalg.solvers;

import com.flag4j.Matrix;
import com.flag4j.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealLstsqSolverTests {
    static final RealLstsqSolver solver = new RealLstsqSolver();

    static double[][] aEntries;
    static Matrix A;


    @Test
    void solveTest() {
        double[] bEntries;
        double[] expEntries;
        Vector b;
        Vector exp;

        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{
                {1, 2, 1.55},
                {1, 2.3, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-54.87309197651661, 27.117873450750153, 1.3144161774298762};
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);

        assertEquals(exp, solver.solve(A, b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[]{1.4, 5.6, -99.35};
        expEntries = new double[]{-0.5953239804326732, -3.745207845180096, 1.2152364890706673};
        A = new Matrix(aEntries);
        b = new Vector(bEntries);
        exp = new Vector(expEntries);

        assertEquals(exp, solver.solve(A, b));
    }


    @Test
    void solveTestMatrix() {
        double[][] bEntries, expEntries;
        Matrix B, exp;

        // ----------------- Sub-case 1 -----------------
        aEntries = new double[][]{
                {1, 2, 1.55},
                {1, 2.3, -1.444},
                {1, 2, -75.1}};
        bEntries = new double[][]{
                {1.4, 90.2},
                {5.6, -0.024},
                {-99.35, 2.5}};
        expEntries = new double[][]{
                {-54.87309197651661, 667.0824135681671},
                {27.117873450750153, -289.3279321591651},
                {1.3144161774298762, 1.1441617742987604}};
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        assertEquals(exp, solver.solve(A, B));

        // ----------------- Sub-case 2 -----------------
        aEntries = new double[][]{
                {1.415, 35.6, 111.56},
                {1.14, -2.145, -1.444},
                {1, 2, -75.1},
                {56.4, 0.0024, 1},
                {-1, 35, 6.4}};
        bEntries = new double[][]{
                {1.4, 5.6, -99.35},
                {90.2, -0.024, 2.5},
                {20024.5, -9.42, 0.024},
                {25.6, 2.0, -895.2},
                {1.34, 245.006, 0.0345115}
        };
        expEntries = new double[][]{
                {11.236812466218089, -0.025918271804249018, -15.869733971321157},
                {235.3971728868125, 5.076829738809294, -0.6842081525682759},
                {-135.59214856020083, -0.9734072210388273, -0.3934543149083684}
        };
        A = new Matrix(aEntries);
        B = new Matrix(bEntries);
        exp = new Matrix(expEntries);

        assertEquals(exp, solver.solve(A, B));
    }
}
