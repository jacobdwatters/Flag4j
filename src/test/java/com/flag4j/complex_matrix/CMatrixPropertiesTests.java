package com.flag4j.complex_matrix;

import com.flag4j.CMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CMatrixPropertiesTests {

    CNumber[][] entriesA;
    CMatrix A;
    boolean expBoolResult;

    @Test
    void isIdentityTestCase() {
        // --------------- Sub-case 1 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2, 123.45), new CNumber(3, -4.551)},
                {new CNumber(-0.442, 12.34), new CNumber(13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(90, -43.18)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 2 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2, 123.45), new CNumber(3, -4.551)},
                {new CNumber(-0.442, 12.34), new CNumber(13.5), new CNumber(35.6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 3 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1, 3.1335), new CNumber(2, 566.72)},
                {new CNumber(-0.442, 67.105), new CNumber(13.5, -78.431)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 4 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 5 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 6 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = true;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 7 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1), new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(0), new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 8 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 9 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0, 0.3232), new CNumber(0)},
                {new CNumber(0), new CNumber(1), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());

        // --------------- Sub-case 10 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(1, -5), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(1)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertEquals(expBoolResult, A.isI());
    }


    @Test
    void isRealTestCase() {
        // --------------- Sub-case 1 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2, 123.45), new CNumber(3, -4.551)},
                {new CNumber(-0.442, 12.34), new CNumber(13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(90, -43.18)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());


        // --------------- Sub-case 2 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isReal());

        // --------------- Sub-case 2 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(13.5,1.4), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());

        // --------------- Sub-case 3 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(-13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(0, 1.90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isReal());
    }


    @Test
    void isComplexTestCase() {
        // --------------- Sub-case 1 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2, 123.45)},
                {new CNumber(-0.442, 12.34), new CNumber(13.5)},
                {new CNumber(0.4441), new CNumber(6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());


        // --------------- Sub-case 2 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertFalse(A.isComplex());

        // --------------- Sub-case 2 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(13.5,1.4), new CNumber(35.6)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());

        // --------------- Sub-case 3 ---------------
        entriesA = new CNumber[][]{
                {new CNumber(1), new CNumber(2), new CNumber(3)},
                {new CNumber(-0.442), new CNumber(-13.5), new CNumber(35.6)},
                {new CNumber(0.4441), new CNumber(6), new CNumber(0, 1.90)}};
        A = new CMatrix(entriesA);
        expBoolResult = false;
        assertTrue(A.isComplex());
    }
}
