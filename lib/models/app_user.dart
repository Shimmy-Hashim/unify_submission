import 'package:cloud_firestore/cloud_firestore.dart';

/// Firestore users/{uid} document (Section 8 of the project brief).
/// Age is always computed from dateOfBirth — never stored.
class AppUser {
  const AppUser({
    required this.uid,
    required this.name,
    required this.email,
    required this.dateOfBirth,
    required this.dependants,
    this.createdAt,
    this.lastActive,
  });

  final String uid;
  final String name;
  final String email;
  final DateTime dateOfBirth;

  final int dependants;
  final DateTime? createdAt;
  final DateTime? lastActive;

  int get age {
    final now = DateTime.now();
    var age = now.year - dateOfBirth.year;
    if (now.month < dateOfBirth.month ||
        (now.month == dateOfBirth.month && now.day < dateOfBirth.day)) {
      age--;
    }
    return age;
  }

  factory AppUser.fromDoc(DocumentSnapshot<Map<String, dynamic>> doc) {
    final data = doc.data()!;
    return AppUser(
      uid: doc.id,
      name: data['name'] as String? ?? '',
      email: data['email'] as String? ?? '',
      dateOfBirth: (data['date_of_birth'] as Timestamp).toDate(),
      dependants: (data['dependants'] as num?)?.toInt() ?? 0,
      createdAt: (data['created_at'] as Timestamp?)?.toDate(),
      lastActive: (data['last_active'] as Timestamp?)?.toDate(),
    );
  }

  /// Fields written at account creation. created_at / last_active are set
  /// with server timestamps by FirebaseService, not here.
  Map<String, dynamic> toCreateMap() => {
        'name': name,
        'email': email,
        'date_of_birth': Timestamp.fromDate(dateOfBirth),
        'dependants': dependants,
      };
}
