//! Tests for big-integer and bit conversion functions.
//!
//! Covers roundtrips, consistency between u64/i64 and string-based paths,
//! edge cases, adversarial patterns (powers of 2, Fibonacci, all-nines),
//! and precondition enforcement.

use _rsqlt::fastsim::vm::{
    bits_to_int, bits_to_int_str, bits_to_uint, bits_to_uint_str, decimal_str_to_bits, int_to_bits,
    signed_decimal_str_to_bits,
};

// ============================================================
// 1. Small integer roundtrips (int_to_bits / bits_to_int / bits_to_uint)
// ============================================================

#[test]
fn test_int_to_bits_zero() {
    let bits = int_to_bits(0, 8);
    assert_eq!(bits, vec![false; 8]);
}

#[test]
fn test_int_to_bits_max_unsigned_8bit() {
    let bits = int_to_bits(255, 8);
    assert_eq!(bits, vec![true; 8]);
}

#[test]
fn test_int_to_bits_negative() {
    let bits = int_to_bits(-1, 8);
    assert_eq!(bits, vec![true; 8]);
}

#[test]
fn test_int_to_bits_min_signed_8bit() {
    let bits = int_to_bits(-128, 8);
    let mut expected = vec![false; 8];
    expected[0] = true;
    assert_eq!(bits, expected);
}

#[test]
fn test_int_to_bits_1bit() {
    assert_eq!(int_to_bits(0, 1), vec![false]);
    assert_eq!(int_to_bits(1, 1), vec![true]);
}

#[test]
fn test_bits_roundtrip_unsigned() {
    for val in 0u64..256 {
        let bits = int_to_bits(val as i64, 8);
        let back = bits_to_uint(&bits);
        assert_eq!(back, val, "Unsigned roundtrip failed for {}", val);
    }
}

#[test]
fn test_bits_roundtrip_signed() {
    for val in -128i64..128 {
        let bits = int_to_bits(val, 8);
        let back = bits_to_int(&bits);
        assert_eq!(back, val, "Signed roundtrip failed for {}", val);
    }
}

#[test]
fn test_bits_to_int_empty() {
    assert_eq!(bits_to_int(&[]), 0);
}

#[test]
fn test_bits_single_bit() {
    assert_eq!(bits_to_uint(&[false]), 0);
    assert_eq!(bits_to_uint(&[true]), 1);
    assert_eq!(bits_to_int(&[false]), 0);
    assert_eq!(bits_to_int(&[true]), -1);
}

// ============================================================
// 2. Unsigned string roundtrips (decimal_str_to_bits / bits_to_uint_str)
// ============================================================

#[test]
fn roundtrip_uint_zero_various_widths() {
    for &width in &[1, 8, 64, 256, 2048] {
        let bits = decimal_str_to_bits("0", width);
        assert_eq!(bits.len(), width, "width={}", width);
        assert_eq!(bits_to_uint_str(&bits), "0", "width={}", width);
    }
}

#[test]
fn roundtrip_uint_small_values() {
    for &(val_str, width) in &[("1", 8), ("255", 8), ("256", 16), ("65535", 16)] {
        let bits = decimal_str_to_bits(val_str, width);
        let back = bits_to_uint_str(&bits);
        assert_eq!(back, val_str, "Failed roundtrip for {}", val_str);
    }
}

#[test]
fn roundtrip_uint_powers_of_2() {
    let pows: Vec<(String, usize)> = vec![
        ("1".into(), 8),
        ("2".into(), 8),
        ("4".into(), 8),
        ("128".into(), 8),
        ("256".into(), 16),
        ("1024".into(), 16),
    ];
    for (val_str, width) in &pows {
        let bits = decimal_str_to_bits(val_str, *width);
        let back = bits_to_uint_str(&bits);
        assert_eq!(&back, val_str, "Failed roundtrip for {}", val_str);
    }
}

#[test]
fn roundtrip_uint_2pow128() {
    let val = "340282366920938463463374607431768211456";
    let bits = decimal_str_to_bits(val, 256);
    let back = bits_to_uint_str(&bits);
    assert_eq!(back, val, "Failed roundtrip for 2^128");
}

#[test]
fn roundtrip_uint_2pow100_in_2048() {
    let val = "1267650600228229401496703205376";
    let bits = decimal_str_to_bits(val, 2048);
    let back = bits_to_uint_str(&bits);
    assert_eq!(back, val, "Failed roundtrip for 2^100 in 2048 bits");
}

// ============================================================
// 3. Signed string roundtrips (signed_decimal_str_to_bits / bits_to_int_str)
// ============================================================

#[test]
fn roundtrip_signed_neg1_various_widths() {
    for &width in &[8, 16, 64, 2048] {
        let bits = signed_decimal_str_to_bits("-1", width);
        assert_eq!(bits.len(), width, "width={}", width);
        let back = bits_to_int_str(&bits);
        assert_eq!(back, "-1", "width={}: got {}", width, back);
    }
}

#[test]
fn roundtrip_signed_neg5_8bit() {
    let bits = signed_decimal_str_to_bits("-5", 8);
    assert_eq!(bits_to_int_str(&bits), "-5");
}

#[test]
fn roundtrip_signed_neg128_8bit() {
    let bits = signed_decimal_str_to_bits("-128", 8);
    assert_eq!(bits_to_int_str(&bits), "-128");
}

#[test]
fn roundtrip_signed_pos127_8bit() {
    let bits = signed_decimal_str_to_bits("127", 8);
    assert_eq!(bits_to_int_str(&bits), "127");
}

#[test]
fn roundtrip_signed_zero() {
    let bits = signed_decimal_str_to_bits("0", 8);
    assert_eq!(bits_to_int_str(&bits), "0");
}

#[test]
fn roundtrip_signed_neg_zero() {
    let bits = signed_decimal_str_to_bits("-0", 8);
    assert_eq!(bits_to_int_str(&bits), "0");
}

// ============================================================
// 4. Consistency: string functions vs u64/i64 functions
// ============================================================

#[test]
fn consistency_uint_str_vs_u64_all_8bit() {
    for val in 0u64..256 {
        let bits = int_to_bits(val as i64, 8);
        let str_result = bits_to_uint_str(&bits);
        let u64_result = bits_to_uint(&bits);
        assert_eq!(
            str_result,
            u64_result.to_string(),
            "Mismatch at val={}",
            val
        );
    }
}

#[test]
fn consistency_int_str_vs_i64_all_8bit() {
    for val in -128i64..128 {
        let bits = int_to_bits(val, 8);
        let str_result = bits_to_int_str(&bits);
        let i64_result = bits_to_int(&bits);
        assert_eq!(
            str_result,
            i64_result.to_string(),
            "Mismatch at val={}",
            val
        );
    }
}

#[test]
fn consistency_decimal_str_to_bits_vs_int_to_bits_all_8bit() {
    for val in 0u64..256 {
        let from_str = decimal_str_to_bits(&val.to_string(), 8);
        let from_int = int_to_bits(val as i64, 8);
        assert_eq!(from_str, from_int, "Mismatch at val={}", val);
    }
}

#[test]
fn consistency_signed_str_vs_int_to_bits_all_8bit() {
    for val in -128i64..128 {
        let from_str = signed_decimal_str_to_bits(&val.to_string(), 8);
        let from_int = int_to_bits(val, 8);
        assert_eq!(from_str, from_int, "Mismatch at val={}", val);
    }
}

#[test]
fn consistency_uint_str_roundtrip_16bit() {
    for val in 0u64..65536 {
        let bits = decimal_str_to_bits(&val.to_string(), 16);
        let back = bits_to_uint_str(&bits);
        assert_eq!(
            back,
            val.to_string(),
            "Unsigned 16-bit roundtrip failed for {}",
            val
        );
    }
}

#[test]
fn consistency_int_str_roundtrip_16bit() {
    for val in -32768i64..32768 {
        let bits = signed_decimal_str_to_bits(&val.to_string(), 16);
        let back = bits_to_int_str(&bits);
        assert_eq!(
            back,
            val.to_string(),
            "Signed 16-bit roundtrip failed for {}",
            val
        );
    }
}

// ============================================================
// 5. Edge cases
// ============================================================

#[test]
fn edge_leading_zeros() {
    let bits_007 = decimal_str_to_bits("007", 8);
    let bits_7 = decimal_str_to_bits("7", 8);
    assert_eq!(bits_007, bits_7, "Leading zeros should be ignored");
}

#[test]
fn edge_very_large_number_300_digits() {
    let mut val_str = "1".to_string();
    for _ in 0..300 {
        val_str.push('0');
    }
    let bits = decimal_str_to_bits(&val_str, 2048);
    let back = bits_to_uint_str(&bits);
    assert_eq!(back, val_str, "Very large number roundtrip failed");
}

#[test]
#[should_panic(expected = "exceeds maximum value for bit width")]
fn edge_overflow_256_in_8bits() {
    decimal_str_to_bits("256", 8);
}

#[test]
#[should_panic(expected = "exceeds maximum value for bit width")]
fn edge_overflow_257_in_8bits() {
    decimal_str_to_bits("257", 8);
}

#[test]
fn edge_2_pow_64_in_128_bits() {
    let val = "18446744073709551616";
    let bits = decimal_str_to_bits(val, 128);
    let back = bits_to_uint_str(&bits);
    assert_eq!(back, val);
}

#[test]
fn edge_max_u64_in_64_bits() {
    let val = "18446744073709551615";
    let bits = decimal_str_to_bits(val, 64);
    let back = bits_to_uint_str(&bits);
    assert_eq!(back, val);
}

#[test]
fn edge_signed_neg1_is_all_ones() {
    for &width in &[8, 16, 64, 256] {
        let bits = signed_decimal_str_to_bits("-1", width);
        assert!(
            bits.iter().all(|&b| b),
            "-1 in {} bits should be all ones",
            width
        );
    }
}

#[test]
fn edge_all_ones_various_widths() {
    for &width in &[1, 8, 16, 32] {
        let bits = vec![true; width];
        let expected = (1u64 << width) - 1;
        let result = bits_to_uint_str(&bits);
        assert_eq!(result, expected.to_string(), "All-ones for width={}", width);
    }
}

#[test]
fn edge_single_bit_values() {
    assert_eq!(bits_to_uint_str(&decimal_str_to_bits("0", 1)), "0");
    assert_eq!(bits_to_uint_str(&decimal_str_to_bits("1", 1)), "1");
}

#[test]
fn edge_empty_bits() {
    let bits: Vec<bool> = vec![];
    assert_eq!(bits_to_uint_str(&bits), "0");
    assert_eq!(bits_to_int_str(&bits), "0");
}

#[test]
fn edge_min_negative_8bit_roundtrip() {
    let bits = signed_decimal_str_to_bits("-128", 8);
    assert_eq!(bits_to_int_str(&bits), "-128");
}

// ============================================================
// 6. bits_to_int_str min values
// ============================================================

#[test]
fn bits_to_int_str_min_8bit() {
    let mut bits = vec![false; 8];
    bits[0] = true;
    assert_eq!(bits_to_int_str(&bits), "-128");
}

#[test]
fn bits_to_int_str_min_16bit() {
    let mut bits = vec![false; 16];
    bits[0] = true;
    assert_eq!(bits_to_int_str(&bits), "-32768");
}

// ============================================================
// 7. Large roundtrips (2048-bit)
// ============================================================

#[test]
fn roundtrip_2048_bit_max_positive() {
    let mut bits = vec![true; 2048];
    bits[0] = false;
    let result = bits_to_int_str(&bits);
    assert!(!result.starts_with('-'), "2^2047-1 should be positive");
    let bits2 = decimal_str_to_bits(&result, 2048);
    let back = bits_to_uint_str(&bits2);
    assert_eq!(back, result, "Roundtrip for 2^2047-1 failed");
}

#[test]
fn roundtrip_signed_neg1_2048() {
    let bits = signed_decimal_str_to_bits("-1", 2048);
    assert_eq!(bits.len(), 2048);
    assert!(
        bits.iter().all(|&b| b),
        "-1 in 2048 bits should be all ones"
    );
    assert_eq!(bits_to_int_str(&bits), "-1");
}

// ============================================================
// 8. Adversarial: powers of 2 up to 2^200
// ============================================================

#[test]
fn adversarial_all_powers_of_2_up_to_200() {
    let mut val_digits: Vec<u8> = vec![1];
    for k in 0..=200 {
        let val_str: String = val_digits
            .iter()
            .rev()
            .map(|d| (b'0' + d) as char)
            .collect();
        let bits = decimal_str_to_bits(&val_str, 256);
        let back = bits_to_uint_str(&bits);
        assert_eq!(back, val_str, "Power-of-2 roundtrip failed for 2^{}", k);
        let mut carry: u8 = 0;
        for d in val_digits.iter_mut() {
            let v = *d * 2 + carry;
            *d = v % 10;
            carry = v / 10;
        }
        if carry > 0 {
            val_digits.push(carry);
        }
    }
}

// ============================================================
// 9. Adversarial: all-ones patterns
// ============================================================

#[test]
fn adversarial_all_ones_patterns() {
    for k in 1..=64 {
        let bits = vec![true; k];
        let result = bits_to_uint_str(&bits);
        if k <= 63 {
            let expected = ((1u64 << k) - 1).to_string();
            assert_eq!(result, expected, "All-ones for k={}", k);
        }
        if k == 64 {
            assert_eq!(result, "18446744073709551615", "All-ones for k=64");
        }
    }
}

// ============================================================
// 10. Adversarial: signed boundary values
// ============================================================

#[test]
fn adversarial_signed_boundaries() {
    let widths: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8, 16, 32];
    for &w in &widths {
        let max_pos = if w == 1 { 0i64 } else { (1i64 << (w - 1)) - 1 };
        let min_neg = -(1i64 << (w - 1));

        let bits_max = signed_decimal_str_to_bits(&max_pos.to_string(), w);
        assert_eq!(
            bits_to_int_str(&bits_max),
            max_pos.to_string(),
            "Max positive for w={}",
            w
        );

        let bits_min = signed_decimal_str_to_bits(&min_neg.to_string(), w);
        assert_eq!(
            bits_to_int_str(&bits_min),
            min_neg.to_string(),
            "Min negative for w={}",
            w
        );
    }
}

// ============================================================
// 11. Adversarial: 1-bit signed
// ============================================================

#[test]
fn adversarial_1bit_signed() {
    assert_eq!(bits_to_int_str(&signed_decimal_str_to_bits("0", 1)), "0");
    assert_eq!(bits_to_int_str(&signed_decimal_str_to_bits("-1", 1)), "-1");
}

// ============================================================
// 12. Adversarial: carry propagation
// ============================================================

#[test]
fn adversarial_carry_propagation() {
    for &w in &[8, 16, 32, 64, 128, 256] {
        let bits = signed_decimal_str_to_bits("-1", w);
        assert!(bits.iter().all(|&b| b), "-1 in {} bits: not all ones", w);
    }
}

// ============================================================
// 13. Adversarial: large signed negative
// ============================================================

#[test]
fn adversarial_large_signed_negative() {
    let magnitude = "170141183460469231731687303715884105728";
    let neg_val = format!("-{}", magnitude);
    let bits = signed_decimal_str_to_bits(&neg_val, 128);
    assert_eq!(bits.len(), 128);
    assert!(bits[0], "MSB should be 1 for negative");
    for (i, bit) in bits.iter().enumerate().take(128).skip(1) {
        assert!(!bit, "Bit {} should be 0 for min negative 128-bit", i);
    }
    assert_eq!(
        bits_to_int_str(&bits),
        neg_val,
        "Large negative roundtrip failed"
    );
}

// ============================================================
// 14. Adversarial: Fibonacci roundtrip
// ============================================================

#[test]
fn adversarial_fibonacci_roundtrip() {
    let mut a: Vec<u8> = vec![0];
    let mut b: Vec<u8> = vec![1];
    for _i in 0..200 {
        let mut c = Vec::new();
        let mut carry: u8 = 0;
        let max_len = a.len().max(b.len());
        for j in 0..max_len {
            let da = if j < a.len() { a[j] } else { 0 };
            let db = if j < b.len() { b[j] } else { 0 };
            let sum = da + db + carry;
            c.push(sum % 10);
            carry = sum / 10;
        }
        if carry > 0 {
            c.push(carry);
        }
        let val_str: String = c.iter().rev().map(|d| (b'0' + d) as char).collect();
        let bits = decimal_str_to_bits(&val_str, 1024);
        let back = bits_to_uint_str(&bits);
        assert_eq!(back, val_str, "Fibonacci roundtrip failed for F({}+2)", _i);
        a = b;
        b = c;
    }
}

// ============================================================
// 15. Adversarial: all-nines patterns
// ============================================================

#[test]
fn adversarial_all_nines() {
    for num_nines in 1..=20 {
        let val_str: String = "9".repeat(num_nines);
        let bits = decimal_str_to_bits(&val_str, 128);
        let back = bits_to_uint_str(&bits);
        assert_eq!(
            back, val_str,
            "All-nines roundtrip failed for {} nines",
            num_nines
        );
    }
}

// ============================================================
// 16. Powers of 10
// ============================================================

#[test]
fn powers_of_10_roundtrip() {
    let mut val = "1".to_string();
    for _ in 0..20 {
        val.push('0');
        let bits = decimal_str_to_bits(&val, 128);
        let back = bits_to_uint_str(&bits);
        assert_eq!(back, val, "Power-of-10 roundtrip failed for {}", val);
    }
}

// ============================================================
// 17. Boundary values at bit limits
// ============================================================

#[test]
fn boundary_values_at_bit_limits() {
    let test_cases: Vec<(String, usize, &str)> = vec![
        ("127".into(), 8, "127"),
        ("128".into(), 8, "128"),
        ("255".into(), 8, "255"),
        ("32767".into(), 16, "32767"),
        ("32768".into(), 16, "32768"),
        ("65535".into(), 16, "65535"),
    ];
    for (input, width, expected) in &test_cases {
        let bits = decimal_str_to_bits(input, *width);
        let back = bits_to_uint_str(&bits);
        assert_eq!(
            &back, expected,
            "Boundary test for {} width={}",
            input, width
        );
    }
}

// ============================================================
// 18. Performance: large bit width roundtrip
// ============================================================

#[test]
fn perf_large_width_roundtrip() {
    let mut bits = vec![false; 4096];
    bits[0] = true;
    let result = bits_to_uint_str(&bits);
    let bits2 = decimal_str_to_bits(&result, 4096);
    assert_eq!(bits, bits2, "4096-bit power-of-2 roundtrip failed");
}

// ============================================================
// 19. Precondition enforcement (panics on oversize inputs)
// ============================================================

#[test]
#[should_panic(expected = "exceeds i64 capacity (max 63)")]
fn precondition_int_to_bits_max_width() {
    int_to_bits(0, 64);
}

#[test]
#[should_panic(expected = "exceeds u64 capacity (max 64)")]
fn precondition_bits_to_uint_max_width() {
    bits_to_uint(&[false; 65]);
}

#[test]
#[should_panic(expected = "exceeds i64 capacity (max 63)")]
fn precondition_bits_to_int_max_width() {
    bits_to_int(&[false; 64]);
}

#[test]
#[should_panic(expected = "Invalid digit in decimal string")]
fn precondition_invalid_decimal_string() {
    decimal_str_to_bits("123abc", 8);
}

// ============================================================
// 20. 4096-bit formatting
// ============================================================

#[test]
fn formatting_4096bit_no_overflow() {
    let bits = decimal_str_to_bits("123456789012345678901234567890", 4096);
    let s = bits_to_int_str(&bits);
    assert_eq!(s, "123456789012345678901234567890");
}
