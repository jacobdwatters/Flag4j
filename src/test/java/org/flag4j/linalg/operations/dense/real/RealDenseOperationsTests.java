package org.flag4j.linalg.operations.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.linalg.operations.common.real.RealOperations;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.operations.dense.real.RealDenseOperations.prod;
import static org.junit.jupiter.api.Assertions.*;

class RealDenseOperationsTests {
    double[] src1, src2;
    double[] expResult;
    double expResultD;
    Shape shape1;
    Shape shape2;
    double a;

    @Test
    void addTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1+9.1233, 0.98332+22, 134.556+0.00009234,
                -9.13+11.234, -100.234-88.1, 0.0000000004+13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertArrayEquals(expResult, add(shape1, src1, shape2, src2));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(LinearAlgebraException.class, () -> add(shape1, src1, shape2, src2));

        // ---------- Sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);

        assertThrows(LinearAlgebraException.class, () -> add(shape1, src1, shape2, src2));
    }


    @Test
    void addScalarTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1+a, 0.98332+a, 134.556+a,
                -9.13+a, -100.234+a, 0.0000000004+a};

        assertArrayEquals(expResult, add(src1, a));
    }


    @Test
    void subTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4};
        expResult = new double[]{1-9.1233, 0.98332-22, 134.556-0.00009234,
                -9.13-11.234, -100.234+88.1, 0.0000000004-13.4};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234, -88.1, 13.4, 1, 1};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        src2 = new double[]{9.1233, 22, 0.00009234, 11.234};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> sub(src1, shape1, src2, shape2));
    }


    @Test
    void subScalarTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1-a, 0.98332-a, 134.556-a,
                -9.13-a, -100.234-a, 0.0000000004-a};

        assertArrayEquals(expResult, sub(src1, a));
    }


    @Test
    void prodTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.8000000004};
        expResultD = 1*0.98332*134.556*-9.13*-100.234*0.8000000004;
        assertEquals(expResultD, prod(src1));

        // ---------- Sub-case 2 -----------------
        src1 = new double[]{};
        expResultD = 0;
        assertEquals(expResultD, prod(src1));
    }


    @Test
    void scaleMultTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new double[]{1, 0.98332, 134.556, -9.13, -100.234, 0.0000000004};
        a = 11.93;
        expResult = new double[]{1*a, 0.98332*a, 134.556*a,
                -9.13*a, -100.234*a, 0.0000000004*a};

        assertArrayEquals(expResult, RealOperations.scalMult(src1, a));
    }
}
