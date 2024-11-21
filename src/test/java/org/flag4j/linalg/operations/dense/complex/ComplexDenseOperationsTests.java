package org.flag4j.linalg.operations.dense.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.operations.common.semiring_ops.AggregateSemiRing;
import org.flag4j.linalg.operations.common.semiring_ops.SemiRingOperations;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseOperations;
import org.flag4j.linalg.operations.dense.semiring_ops.DenseSemiringOperations;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ComplexDenseOperationsTests {

    Complex128[] src1, src2;
    Field<Complex128>[] expResult;
    Complex128 expResultC;
    Shape shape1, shape2;
    double a;
    Complex128 aC;


    @Test
    void addTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(-9234.23), new Complex128(109.2234, 1.435),
                new Complex128(0, -1943.134), new Complex128(-9234.1, -3245)};
        expResult = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertArrayEquals(expResult, DenseSemiringOperations.add(src1, shape1, src2, shape2, null));

        // ---------- Sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245), new Complex128(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> DenseSemiringOperations.add(src1, shape1, src2, shape2, null));

        // ---------- Sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
        assertThrows(LinearAlgebraException.class, () -> DenseSemiringOperations.add(src1, shape1, src2, shape2, null));
    }


    @Test
    void addDoubleTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9+a, -1), new Complex128(-0.99+a, 13.445),
                new Complex128(0.9133+a), new Complex128(0+a, 10.3)};
        assertArrayEquals(expResult, RealFieldDenseOperations.add(src1, a));
    }


    @Test
    void addComplex128TestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).add(aC), new Complex128(-0.99, 13.445).add(aC),
                new Complex128(0.9133).add(aC), new Complex128(0, 10.3).add(aC)};
        assertArrayEquals(expResult, SemiRingOperations.add(src1, null, aC));
    }


    @Test
    void subTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(-9234.23), new Complex128(109.2234, 1.435),
                new Complex128(0, -1943.134), new Complex128(-9234.1, -3245)};

        expResult = new Complex128[]{new Complex128(9+9234.23, -1), new Complex128(-0.99-109.2234, 13.445-1.435),
                new Complex128(0.9133, 1943.134), new Complex128(9234.1, 10.3+3245)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertArrayEquals(expResult, DenseFieldOps.sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 2 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435),
                new Complex128(0.9133, -1943.134), new Complex128(-9234.1, 10.3-3245), new Complex128(0, 1)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertThrows(LinearAlgebraException.class, () -> DenseFieldOps.sub(src1, shape1, src2, shape2));

        // ---------- Sub-case 3 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        src2 = new Complex128[]{new Complex128(9-9234.23, -1), new Complex128(-0.99+109.2234, 13.445+1.435)};
        shape1 = new Shape(src1.length);
        shape2 = new Shape(src2.length);
//        assertThrows(LinearAlgebraException.class, () -> DenseFieldOps.sub(src1, shape1, src2, shape2));
    }


    @Test
    void subDoubleTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9-a, -1), new Complex128(-0.99-a, 13.445),
                new Complex128(0.9133-a), new Complex128(0-a, 10.3)};
        assertArrayEquals(expResult, RealFieldDenseOperations.sub(src1, a));
    }


    @Test
    void subComplex128TestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).sub(aC), new Complex128(-0.99, 13.445).sub(aC),
                new Complex128(0.9133).sub(aC), new Complex128(0, 10.3).sub(aC)};
        assertArrayEquals(expResult, SemiRingOperations.sub(src1, null, aC));
    }


    @Test
    void sumTestCase() {
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        expResultC = new Complex128(9-0.99+0.9133, -1+13.445+10.3);
        assertEquals(expResultC, AggregateSemiRing.sum(src1));
    }


    @Test
    void prodTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};

        expResultC = new Complex128(-1147.60574505, 42.660699650000005);
        assertEquals(expResultC, AggregateSemiRing.prod(src1));

        // ---------- Sub-case 2 -----------------
        src1 = new Complex128[]{};
        assertNull(AggregateSemiRing.prod(src1));
    }


    @Test
    void scalMultDoubleTestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        a = 933.1334;
        expResult = new Complex128[]{new Complex128(9*a, -1*a), new Complex128(-0.99*a, 13.445*a),
                new Complex128(0.9133*a), new Complex128(0*a, 10.3*a)};
//        assertArrayEquals(expResult, DenseFieldOps.scalMult(src1, a));
    }


    @Test
    void scalMultComplex128TestCase() {
        // ---------- Sub-case 1 -----------------
        src1 = new Complex128[]{new Complex128(9, -1), new Complex128(-0.99, 13.445),
                new Complex128(0.9133), new Complex128(0, 10.3)};
        aC = new Complex128(10.34, -1.334);
        expResult = new Complex128[]{new Complex128(9, -1).mult(aC), new Complex128(-0.99, 13.445).mult(aC),
                new Complex128(0.9133).mult(aC), new Complex128(0, 10.3).mult(aC)};
        assertArrayEquals(expResult, SemiRingOperations.scalMult(src1, aC));
    }
}
