package org.flag4j.sparse_complex_tensor;

import org.flag4j.arrays_old.sparse.CooCTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CooCTensorGetSetTests {
    static CooCTensor A;
    static Shape aShape;
    static CNumber[] aEntries;
    static int[][] aIndices;

    static CooCTensor exp;
    static Shape expShape;
    static CNumber[] expEntries;
    static int[][] expIndices;


    @Test
    void cooTensorSetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new CNumber[]{
                new CNumber(1, 2), new CNumber(0.2324, -239.1),
                new CNumber(5.1), new CNumber(0, 18.2)};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        // --------------- Sub-case 1 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new CNumber[]{
                new CNumber(1, 2), new CNumber(2.41, -23.23),
                new CNumber(5.1), new CNumber(0, 18.2)};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(new CNumber(2.41, -23.23), 1, 2, 0, 0));

        // --------------- Sub-case 2 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new CNumber[]{
                new CNumber(1, 2), new CNumber(0.2324, -239.1), new CNumber(0, -9.2),
                new CNumber(5.1), new CNumber(0, 18.2)};
        expIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {1, 2, 4, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(new CNumber(0, -9.2), 1, 2, 4, 0));

        // --------------- Sub-case 3 ---------------
        expShape = new Shape(3, 4, 5, 1);
        expEntries = new CNumber[]{
                new CNumber(0.001), new CNumber(1, 2), new CNumber(0.2324, -239.1),
                new CNumber(5.1), new CNumber(0, 18.2)};
        expIndices = new int[][]{{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.set(0.001, 0, 0, 0, 0));

        // --------------- Sub-case 4 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, -1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.set(5, 1, 2, 5, 0));
    }


    @Test
    void cooTensorGetTest() {
        aShape = new Shape(3, 4, 5, 1);
        aEntries = new CNumber[]{
                new CNumber(1, 2), new CNumber(0.2324, -239.1),
                new CNumber(5.1), new CNumber(0, 18.2)};
        aIndices = new int[][]{{0, 1, 0, 0}, {1, 2, 0, 0}, {2, 3, 2, 0}, {2, 3, 4, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        // --------------- Sub-case 1 ---------------
        assertEquals(new CNumber(0.2324, -239.1), A.get(1, 2, 0, 0));

        // --------------- Sub-case 2 ---------------
        assertEquals(CNumber.ZERO, A.get(1, 2, 4, 0));

        // --------------- Sub-case 3 ---------------
        assertEquals(CNumber.ZERO, A.get(0, 0, 0, 0));

        // --------------- Sub-case 4 ---------------
        assertEquals(new CNumber(1, 2), A.get(0, 1, 0, 0));

        // --------------- Sub-case 5 ---------------
        assertEquals(new CNumber(5.1), A.get(2, 3, 2, 0));

        // --------------- Sub-case 6 ---------------
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(-1, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(0, 0, 0, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> A.get(1, 2, 5, 0));
    }
}
