package com.flag4j.linalg.solvers;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.SingularMatrixException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexForwardSolverTests {

    ComplexForwardSolver solver;
    CNumber[][] lEntries;
    CMatrix L;

    @Test
    void solveVectorTestCase() {
        solver = new ComplexForwardSolver();
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(2.45, 15.5), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[]{
                new CNumber(1.345, 2.45), new CNumber(-92.24, 7.24), new CNumber(0, 45.2)
        };
        b = new CVector(bEntries);

        expEntries = new CNumber[]{
                new CNumber("-0.2408177905308465 + 0.177948350071736i"),
                new CNumber("6.742734536376031 + 19.670300024162774i"),
                new CNumber("-6.458765618863107 + 2.169298473450951i")};
        exp = new CVector(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- Sub-case 2 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[]{
                new CNumber(1.345, 2.45), new CNumber(-92.24, 7.24), new CNumber(0, 45.2)
        };
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(SingularMatrixException.class, ()->solver.solve(L, finalB));

        // ---------------------- Sub-case 3 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[]{
                new CNumber(1.345, 2.45), new CNumber(-92.24, 7.24), new CNumber(0, 45.2),
                new CNumber(2.45, -1300.13415)
        };
        b = new CVector(bEntries);

        CVector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveMatrixTestCase() {
        solver = new ComplexForwardSolver();
        CNumber[][] bEntries, expEntries;
        CMatrix b, exp;

        // ---------------------- Sub-case 1 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(2.45, 15.5), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(256.2)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1.345, 2.45), new CNumber(2.4, -.36)},
                {new CNumber(-92.24, 7.24), new CNumber(0, 13.5)},
                {new CNumber(0, 45.2), new CNumber(-1.3567)}
        };
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("-0.2408177905308465 + 0.177948350071736i"), new CNumber("0.0726542324246772 + 0.24964131994261118i")},
                {new CNumber("6.742734536376031 + 19.670300024162774i"), new CNumber("14.250401796793327 - 1.5933241423340474i")},
                {new CNumber("-6.458765618863107 + 2.169298473450951i"), new CNumber("0.3668372528597201 + 4.645626702643428i")}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- Sub-case 2 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(2.45, 15.5), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber()}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1.345, 2.45), new CNumber(2.4, -.36)},
                {new CNumber(-92.24, 7.24), new CNumber(0, 13.5)},
                {new CNumber(0, 45.2), new CNumber(-1.3567)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB = b;
        assertThrows(SingularMatrixException.class, ()->solver.solve(L, finalB));

        // ---------------------- Sub-case 3 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1.25, -9.25), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(2.45, 15.5), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber()}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1.345, 2.45), new CNumber(2.4, -.36)},
                {new CNumber(-92.24, 7.24), new CNumber(0, 13.5)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveUnitVectorTestCase() {
        solver = new ComplexForwardSolver(true);
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // ---------------------- Sub-case 1 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(1), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[]{
                new CNumber(1.345, 2.45), new CNumber(-92.24, 7.24), new CNumber(0, 45.2)
        };
        b = new CVector(bEntries);

        expEntries = new CNumber[]{
                new CNumber("1.345 + 2.45i"),
                new CNumber("1008.1453000000001 + 2003.2728200000001i"),
                new CNumber("-169164.93180900003 + 79027.31987100001i")};
        exp = new CVector(expEntries);

        assertEquals(exp, solver.solve(L, b));

        // ---------------------- Sub-case 2 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(1), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[]{
                new CNumber(1.345, 2.45), new CNumber(-92.24, 7.24),
                new CNumber(0, 45.2), new CNumber(2.45, -1300.13415)
        };
        b = new CVector(bEntries);

        CVector finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }


    @Test
    void solveUnitMatrixTestCase() {
        solver = new ComplexForwardSolver(true);
        CNumber[][] bEntries, expEntries;
        CMatrix b, exp;

        // ---------------------- Sub-case 1 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(1), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1.345, 2.45), new CNumber(2.4, -.36)},
                {new CNumber(-92.24, 7.24), new CNumber(0, 13.5)},
                {new CNumber(0, 45.2), new CNumber(-1.3567)}
        };
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("1.345 + 2.45i"), new CNumber("2.4 - 0.36i")},
                {new CNumber("1008.1453000000001 + 2003.2728200000001i"), new CNumber("1956.68016 - 283.5456i")},
                {new CNumber("-169164.93180900003 + 79027.31987100001i"), new CNumber("18799.102988000002 + 163512.690432i")}
        };
        exp = new CMatrix(expEntries);
        assertEquals(exp, solver.solve(L, b));

        // ---------------------- Sub-case 2 ----------------------
        lEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(), new CNumber()},
                {new CNumber(-815.5, 1.444), new CNumber(1), new CNumber()},
                {new CNumber(0, -9.256), new CNumber(2.45, -83.2), new CNumber(1)}
        };
        L = new CMatrix(lEntries);
        bEntries = new CNumber[][]{
                {new CNumber(1.345, 2.45), new CNumber(2.4, -.36)},
                {new CNumber(-92.24, 7.24), new CNumber(0, 13.5)}
        };
        b = new CMatrix(bEntries);

        CMatrix finalB1 = b;
        assertThrows(IllegalArgumentException.class, ()->solver.solve(L, finalB1));
    }
}
