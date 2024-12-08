package org.flag4j.complex_tensor;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorTransposeTests {

    static int[] aAxes;

    static Complex128[] aEntries;
    static Complex128[] expEntries;

    static Shape aShape;
    static Shape expShape;

    static CTensor A;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(235.61, 1.45), new Complex128(0, -0.00024),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(1.345, -781.2),
                new Complex128(0.014, -2.45),  new Complex128(-140.0),  new Complex128(0, 1.5),
                 new Complex128(51.0, 24.56),  new Complex128(6.1, -0.03),  new Complex128(-0.00014, 1.34)};
        aShape = new Shape(3, 2, 1, 2);
        A = new CTensor(aShape, aEntries);
    }


    @Test
    void transposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(-85.1, 9.234),  new Complex128(0, 1.5),
                new Complex128(0, -0.00024), new Complex128(0.014, -2.45),  new Complex128(6.1, -0.03),
                new Complex128(235.61, 1.45), new Complex128(1.345, -781.2),  new Complex128(51.0, 24.56),
                new Complex128(1.0),  new Complex128(-140.0),  new Complex128(-0.00014, 1.34)};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(0, -0.00024), new Complex128(235.61, 1.45),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(0.014, -2.45),
                new Complex128(1.345, -781.2),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(6.1, -0.03),  new Complex128(51.0, 24.56),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(-85.1, 9.234),  new Complex128(0, 1.5),
                new Complex128(0, -0.00024), new Complex128(0.014, -2.45),  new Complex128(6.1, -0.03),
                new Complex128(235.61, 1.45), new Complex128(1.345, -781.2),  new Complex128(51.0, 24.56),
                new Complex128(1.0),  new Complex128(-140.0),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(0, -0.00024), new Complex128(235.61, 1.45),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(0.014, -2.45),
                new Complex128(1.345, -781.2),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(6.1, -0.03),  new Complex128(51.0, 24.56),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.T(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.T(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.T(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.T(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(LinearAlgebraException.class, ()->A.T(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(LinearAlgebraException.class, ()->A.T(1, 6));
    }


    @Test
    void hermTransposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(-85.1, 9.234),  new Complex128(0, 1.5),
                new Complex128(0, -0.00024), new Complex128(0.014, -2.45),  new Complex128(6.1, -0.03),
                new Complex128(235.61, 1.45), new Complex128(1.345, -781.2),  new Complex128(51.0, 24.56),
                new Complex128(1.0),  new Complex128(-140.0),  new Complex128(-0.00014, 1.34)};
        expShape = new Shape(2, 2, 1, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.H());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(0, -0.00024), new Complex128(235.61, 1.45),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(0.014, -2.45),
                new Complex128(1.345, -781.2),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(6.1, -0.03),  new Complex128(51.0, 24.56),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(3, 1, 2, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.H(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(-85.1, 9.234),  new Complex128(0, 1.5),
                new Complex128(0, -0.00024), new Complex128(0.014, -2.45),  new Complex128(6.1, -0.03),
                new Complex128(235.61, 1.45), new Complex128(1.345, -781.2),  new Complex128(51.0, 24.56),
                new Complex128(1.0),  new Complex128(-140.0),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(2, 1, 2, 3);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.H(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new Complex128[]{
                new Complex128(1.4415, -0.0245), new Complex128(0, -0.00024), new Complex128(235.61, 1.45),
                new Complex128(1.0), new Complex128(-85.1, 9.234), new Complex128(0.014, -2.45),
                new Complex128(1.345, -781.2),  new Complex128(-140.0),  new Complex128(0, 1.5),
                new Complex128(6.1, -0.03),  new Complex128(51.0, 24.56),  new Complex128(-0.00014, 1.34)
        };
        expShape = new Shape(3, 2, 1, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp.conj(), A.H(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.H(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.H(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.H(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.H(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.H(1, 6));
    }
}
